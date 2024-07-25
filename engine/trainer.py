import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import List,Tuple,Dict

def train_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optimizers:torch.optim.Optimizer, devices:str):
    # wandb.watch(model, log_freq=100)
    model.to(devices)
    model.train()
    train_acc,train_loss=0,0
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(devices),y.to(devices)
        y_pred=model(X)
        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()
        y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc +=(y_pred_class==y).sum().item()/len(y_pred)
    train_acc/=len(dataloader)
    train_loss/=len(dataloader)
    return train_acc,train_loss

def test_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module, devices:str):
    model.to(devices)
    model.eval()
    test_loss_values,test_acc_values=0,0
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y=X.to(devices),y.to(devices)
            y_test_pred_logits=model(X)
            
            test_loss=loss_fn(y_test_pred_logits,y)
            test_loss_values+=test_loss.item()
            
            y_pred_class=torch.argmax(y_test_pred_logits,dim=1)
            test_acc_values += ((y_pred_class==y).sum().item()/len(y_test_pred_logits))
        test_loss_values/=len(dataloader)
        test_acc_values/=len(dataloader)
    return test_loss_values,test_acc_values

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 100, early_stopping=None, devices: str="cuda"):
    result = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_acc, train_loss = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizers=optimizer, devices=devices)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, devices=devices)
        
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["test_loss"].append(test_loss)
        result["test_acc"].append(test_acc)
        # wandb.log({"Train Loss": train_loss,
        #            "Test Loss": test_loss,
        #            "Train Accuracy": train_acc,
        #            "Test Accuracy": test_acc,"Epoch":epoch})
        # Check for early stopping
        if early_stopping is not None:
            if early_stopping.step(test_loss):  # You can use any monitored metric here
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return result