import torch
import torch.nn as nn
from torchvision.models import WeightsEnum
from engine import train_model
from typing import Dict, Any

def Plant_Disease_Cls_Model(train_dataloader: torch.utils.data.DataLoader,
                            test_dataloader: torch.utils.data.DataLoader,
                            weights: WeightsEnum,
                            model: torch.nn.Module,
                            class_names: torch.utils.data.DataLoader,
                            learning_rate: float) -> Dict[str, Any]:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Freeze all the base layers in the features section of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # get the lenght our data class
    num_in_features = model.classifier[1].in_features
    num_output_shape  = len(class_names)
    
    # reacreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=num_in_features,
                        out_features=num_output_shape,
                        bias=True)
    ).to(device)

    # nn.CrossEntropyLoss() for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    results = train_model(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      epochs=15,
                      device=device
                      )
    return results
    
    