import torch
import torchvision
from data_loader import create_dataloader
from model import Plant_Disease_Cls_Model
from utils import save_model, plot_loss_curves

# Setup hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/Subsampled_Plant_Dataset/train"
test_dir = "data/Subsampled_Plant_Dataset/test"


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
auto_transforms = weights.transforms()

# create dataloader
train_dataloader, test_dataloder, class_names = create_dataloader(train_dir=train_dir,
                                                                  test_dir=test_dir,
                                                                  transform=auto_transforms,
                                                                  batch_size=BATCH_SIZE)

# train the model
results =Plant_Disease_Cls_Model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloder,
        weights=weights,
        model=model,
        class_names=class_names,
        learning_rate=LEARNING_RATE
    )

# save model
target_dir = "models"
model_name = "Plant disease classifier.pth"
save_model(model=model,
           target_dir=target_dir,
           model_name=model_name)





