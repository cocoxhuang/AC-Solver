from models.models import *
from data.dataset import DualRingActor_dataset, FFN_dataset
from tqdm import tqdm
import os
import argparse
import datetime
import time
import wandb
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train some model on the relator dataset.")

    parser.add_argument("--data_path", type=str, default="pretrain/data/gs_sa.csv",
                        help="Path to the training data file.")
    parser.add_argument("--checkpoint_dir", type=str, default="pretrain/checkpoints",
                        help="Directory to save model checkpoints.")
    
    # model selection
    parser.add_argument("--model", type=str, choices=["FFN", "DualRingActor"], default="FFN",
                        help="Type of model to train.")

    # FFN model hyperparameters
    parser.add_argument("--nodes", type=int, nargs='+', default=[512, 512],
                        help="List of integers specifying the number of nodes in each hidden layer.")
    
    # training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")

    # weight and biases logging
    parser.add_argument("--wandb", type=bool, default=False, help="Enable Weights and Biases logging")

    return parser.parse_args()

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms (PyTorch 1.12+)
    # torch.use_deterministic_algorithms(True, warn_only=True)

def train(args):
    data_path = args.data_path
    checkpoint_dir = args.checkpoint_dir

    model_type = args.model
    lr = args.lr
    batch_size = args.batch_size
    seed = args.seed
    epochs = args.epochs

    # Set all seeds for reproducibility
    set_seed(seed)

    # dataset selection
    print(f"Loading and processing dataset from {data_path}...")
    start_time = time.time()
    if model_type == "FFN":
        datatset = FFN_dataset(data_path)
    elif model_type == "DualRingActor":
        datatset = DualRingActor_dataset(data_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    train_dataloader, eval_dataloader = datatset.dataloaders(
        batch_size=batch_size, seed=seed)
    print(f"Dataset loaded and processed in {time.time() - start_time:.2f}s")
    
    max_relator_length = datatset._max_relator_length
    n_actions = datatset.n_actions
    print(f"Dataset size: {len(datatset._data)}, Max relator length: {max_relator_length}, Actions: {n_actions}")

    # model selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "FFN":
        nodes = args.nodes
        model = FFN(input_size=2*max_relator_length, nodes=nodes, output_size=n_actions)
    elif model_type == "DualRingActor":
        model = DualRingActor(n_actions=n_actions, max_len=2*max_relator_length,
                              head_dim = 32, mlp_dim = 128)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # trainig setup
    criterion = nn.CrossEntropyLoss()
    # TODO: try warming up learning rate scheduler/ other optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # make checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    best_eval_accuracy = 0.0

    # session naming
    session = f"{model_type}_lr{lr}_bs{batch_size}_seed{seed}_"
    session += datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Training session: {session}")

    # weights and biases setup
    if args.wandb:
        run = wandb.init(
            project="ac-solver-pretraining",
            name=session,
            config=vars(args),
            save_code=True,
        )

    # Training loop with timing
    for epoch in tqdm(range(epochs), desc="Training"):
        
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            l = criterion(outputs, batch_y)
            l.backward()
            optimizer.step()
            
            train_loss += l.item()
            num_batches += 1

        # Evaluation
        with torch.no_grad():
            total_correct = 0
            total_samples = 0

            for batch_X, batch_y in eval_dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                # greedy generation
                predicted = torch.argmax(logits, dim=1)
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)

            accuracy = total_correct / total_samples
            avg_train_loss = train_loss / num_batches
            print(f"Epoch {epoch}, Accuracy: {accuracy}, Loss: {avg_train_loss}")

            if accuracy > best_eval_accuracy:
                best_eval_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_{session}.pth"))
                print(f"New best model saved with accuracy: {best_eval_accuracy}")
        
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "eval_accuracy": accuracy,
            })

if __name__ == "__main__":
    args = parse_args()
    train(args)