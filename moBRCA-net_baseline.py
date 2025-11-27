import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MultiOmicsDataset(Dataset):
    def __init__(self, gene_x, methyl_x, mirna_x, y_onehot):
        assert gene_x.shape[0] == methyl_x.shape[0] == mirna_x.shape[0] == y_onehot.shape[0]
        self.gene_x = torch.tensor(gene_x, dtype=torch.float32)
        self.methyl_x = torch.tensor(methyl_x, dtype=torch.float32)
        self.mirna_x = torch.tensor(mirna_x, dtype=torch.float32)

        # one-hot -> class index
        if y_onehot.ndim == 2 and y_onehot.shape[1] > 1:
            y_idx = np.argmax(y_onehot, axis=1)
        else:
            y_idx = y_onehot.squeeze()
        self.y = torch.tensor(y_idx, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (self.gene_x[idx],
                self.methyl_x[idx],
                self.mirna_x[idx],
                self.y[idx])


class OmicsAttention(nn.Module):
    """
    Feature-level attention cho từng omics.
    """
    def __init__(self, n_features, n_embedding=128, n_proj=64, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.n_embedding = n_embedding
        self.n_proj = n_proj

        self.emb = nn.Parameter(torch.randn(n_features, n_embedding))

        self.fc_fx = nn.Linear(n_embedding, n_proj)
        self.bn_fx = nn.BatchNorm1d(n_proj)

        self.fc_a1 = nn.Linear(n_embedding, n_features)
        self.bn_a1 = nn.BatchNorm1d(n_features)
        self.fc_a2 = nn.Linear(n_features, 1)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: (B, n_features)
        return new_rep: (B, n_proj), attn: (B, n_features)
        """
        B, N = x.shape
        assert N == self.n_features

        emb = self.emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, E)
        fe = x.unsqueeze(2) * emb                      # (B, N, E)

        fx = fe.reshape(B * N, self.n_embedding)
        fx = self.fc_fx(fx)
        fx = self.bn_fx(fx)
        fx = self.tanh(fx)
        fx = self.dropout(fx)
        fx = fx.view(B, N, self.n_proj)

        fa1 = fe.reshape(B * N, self.n_embedding)
        fa1 = self.fc_a1(fa1)
        fa1 = self.bn_a1(fa1)
        fa1 = self.tanh(fa1)
        fa1 = self.dropout(fa1)
        fa1 = fa1.view(B, N, self.n_features)

        fa2 = self.fc_a2(fa1)          # (B, N, 1)
        attn_logits = fa2.view(B, N)
        attn = torch.softmax(attn_logits, dim=1)

        new_rep = (attn.unsqueeze(2) * fx).sum(dim=1)  # (B, n_proj)
        return new_rep, attn


class MoBRCANetTorch(nn.Module):
    """
    Phiên bản PyTorch bám sát moBRCA-net gốc:
    - Attention cho gene, methyl, mirna
    - Nối 3 rep -> FC -> Softmax
    """
    def __init__(self,
                 n_gene,
                 n_methyl,
                 n_mirna,
                 n_classes,
                 n_embedding=128,
                 n_proj=64,
                 n_sm_h2=200,
                 dropout=0.2):
        super().__init__()
        self.gene_attn = OmicsAttention(n_gene, n_embedding, n_proj, dropout)
        self.methyl_attn = OmicsAttention(n_methyl, n_embedding, n_proj, dropout)
        self.mirna_attn = OmicsAttention(n_mirna, n_embedding, n_proj, dropout)

        in_dim = 3 * n_proj
        self.fc2 = nn.Linear(in_dim, n_sm_h2)
        self.bn2 = nn.BatchNorm1d(n_sm_h2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(n_sm_h2, n_classes)

    def forward(self, gene_x, methyl_x, mirna_x):
        rep_gene, attn_gene = self.gene_attn(gene_x)
        rep_methyl, attn_methyl = self.methyl_attn(methyl_x)
        rep_mirna, attn_mirna = self.mirna_attn(mirna_x)

        rep_concat = torch.cat([rep_gene, rep_methyl, rep_mirna], dim=1)
        h = self.fc2(rep_concat)
        h = self.bn2(h)
        h = self.elu(h)
        h = self.dropout(h)
        logits = self.fc_out(h)

        return logits, (attn_gene, attn_methyl, attn_mirna)


def train_and_eval(train_ds,
                   test_ds,
                   n_gene,
                   n_methyl,
                   n_mirna,
                   n_classes,
                   res_dir,
                   batch_size=136,
                   epochs=200,
                   lr=1e-2,
                   dropout=0.2,
                   weight_decay=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    model = MoBRCANetTorch(
        n_gene=n_gene,
        n_methyl=n_methyl,
        n_mirna=n_mirna,
        n_classes=n_classes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    best_pred = None
    best_label = None
    best_attn_gene = None
    best_attn_methyl = None
    best_attn_mirna = None

    os.makedirs(res_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for gene_x, methyl_x, mirna_x, y in train_loader:
            gene_x = gene_x.to(device)
            methyl_x = methyl_x.to(device)
            mirna_x = mirna_x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(gene_x, methyl_x, mirna_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # --- eval trên test ---
        model.eval()
        with torch.no_grad():
            for gene_x, methyl_x, mirna_x, y in test_loader:
                gene_x = gene_x.to(device)
                methyl_x = methyl_x.to(device)
                mirna_x = mirna_x.to(device)
                y = y.to(device)

                logits, (attn_gene, attn_methyl, attn_mirna) = model(
                    gene_x, methyl_x, mirna_x
                )
                preds = logits.argmax(dim=1)
                correct = (preds == y).sum().item()
                total = y.size(0)
                test_acc = correct / total

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_pred = preds.cpu().numpy()
                    best_label = y.cpu().numpy()
                    best_attn_gene = attn_gene.cpu().numpy()
                    best_attn_methyl = attn_methyl.cpu().numpy()
                    best_attn_mirna = attn_mirna.cpu().numpy()

        print(
            f"Epoch {epoch:04d} | "
            f"Train loss: {train_loss:.6f}, Train acc: {train_acc:.6f}, "
            f"Best Test acc: {best_acc:.6f}"
        )

    # --- lưu kết quả ---
    if best_pred is not None:
        np.savetxt(os.path.join(res_dir, "prediction.csv"),
                   best_pred, fmt="%.0f", delimiter=",")
        np.savetxt(os.path.join(res_dir, "label.csv"),
                   best_label, fmt="%.0f", delimiter=",")
        np.savetxt(os.path.join(res_dir, "attn_score_gene.csv"),
                   best_attn_gene, fmt="%.6f", delimiter=",")
        np.savetxt(os.path.join(res_dir, "attn_score_methyl.csv"),
                   best_attn_methyl, fmt="%.6f", delimiter=",")
        np.savetxt(os.path.join(res_dir, "attn_score_mirna.csv"),
                   best_attn_mirna, fmt="%.6f", delimiter=",")
        print(f"Saved outputs to {res_dir} | Final best acc: {best_acc:.6f}")
    else:
        print("Warning: no best_pred recorded; check training configuration.")


def main():
    if len(sys.argv) != 9:
        print("Usage: python moBRCA_net_torch.py "
              "train_X.csv train_Y.csv test_X.csv test_Y.csv "
              "num_gene num_cpg num_mirna resDir")
        sys.exit(1)

    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    test_y_path = sys.argv[4]
    n_gene = int(sys.argv[5])
    n_methyl = int(sys.argv[6])
    n_mirna = int(sys.argv[7])
    res_dir = sys.argv[8]

    x_train = pd.read_csv(train_x_path, dtype=np.float32)
    x_test = pd.read_csv(test_x_path, dtype=np.float32)
    y_train = pd.read_csv(train_y_path, dtype=np.float32).values
    y_test = pd.read_csv(test_y_path, dtype=np.float32).values

    x_gene_train = x_train.iloc[:, :n_gene].values
    x_gene_test = x_test.iloc[:, :n_gene].values

    x_methyl_train = x_train.iloc[:, n_gene:n_gene + n_methyl].values
    x_methyl_test = x_test.iloc[:, n_gene:n_gene + n_methyl].values

    x_mirna_train = x_train.iloc[:, n_gene + n_methyl:].values
    x_mirna_test = x_test.iloc[:, n_gene + n_methyl:].values

    n_classes = y_train.shape[1]

    print("# gene:", n_gene,
          "# cpg:", n_methyl,
          "# mirna:", n_mirna,
          "# classes:", n_classes,
          "# train sample:", x_gene_train.shape[0])

    train_ds = MultiOmicsDataset(x_gene_train, x_methyl_train, x_mirna_train, y_train)
    test_ds = MultiOmicsDataset(x_gene_test, x_methyl_test, x_mirna_test, y_test)

    epochs = int(os.getenv("EPOCHS", "200"))
    batch_size = int(os.getenv("BATCH_SIZE", "136"))
    lr = float(os.getenv("LR", "1e-2"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-4"))

    train_and_eval(
        train_ds=train_ds,
        test_ds=test_ds,
        n_gene=n_gene,
        n_methyl=n_methyl,
        n_mirna=n_mirna,
        n_classes=n_classes,
        res_dir=res_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        dropout=0.2,
        weight_decay=weight_decay
    )


if __name__ == "__main__":
    main()
