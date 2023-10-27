import os
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms, datasets,models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            transformed_image = transform(f)
            ax[1].imshow(transformed_image.permute(1, 2, 0))
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {os.path.basename(os.path.dirname(image_path))}", fontsize=16)
    

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

# Path to the main folder containing subfolders for each class
main_folder = "C:/Users/rohan/Downloads/Training dataset task_2B"

# Define the classes (subfolder names)
classes = ["combat","destroyedbuilding","fire", "humanitarianaid", "militaryvehicles" ]
class_to_idx = {classes: i for i, classes in enumerate(classes)}

# Dictionary to store images for each class
class_images = {cls: [] for cls in classes}

# Load images from each class
for cls in classes:
    class_folder = os.path.join(main_folder, cls)

    # Iterate over the files in the folder
    for filename in os.listdir(class_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image file
            img_path = os.path.join(class_folder, filename)
            class_images[cls].append(img_path)

# Define the image paths for each class
image_path_list = {cls: images for cls, images in class_images.items()}

# Call the function to plot transformed images for each class
for cls in classes:
    print(f"Plotting transformed images for class: {cls}")
    plot_transformed_images(image_path_list[cls], transform=data_transform, n=3)


# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root="C:/Users/rohan/Downloads/Training dataset task_2B", # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)



print(f"Train data:\n{train_data}")


# Get class names as a list
class_names = train_data.classes

print(class_names)

# Can also get class names as a dict
class_dict = train_data.class_to_idx
print(class_dict)

# Check the lengths
print("length of training data:",len(train_data))


#Our images are now in the form of a tensor (with shape [3, 64, 64]) 
# and the labels are in the form of an integer relating to a specific class (as referenced by the class_to_idx attribute).


# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, 
                              shuffle=True) 


print(train_dataloader)

for i,(img,label)in enumerate(train_dataloader):
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")




# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final classification layer
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# Modify the final classification layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Send the model to the device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tune the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()


    running_loss = 0.0
    corrects = 0

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_data)
    epoch_acc = corrects.double() / len(train_data)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_model.pth")


#testing phase

# Load the image you want to classify
image_path = "C:/Users/rohan/Downloads/Testing dataset task_2B/building1.jpeg"  # Replace with the path to your test image
image = Image.open(image_path)

# Define the same transform you used for training
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# Apply the transform to your image
transformed_image = data_transform(image)

# Expand the dimensions of the image to make it compatible with the model
transformed_image = transformed_image.unsqueeze(0)

# Load the fine-tuned model
model = models.resnet18(pretrained=False)  # Create a new instance of the model
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Modify the final classification layer
model.load_state_dict(torch.load("fine_tuned_model.pth"))  # Load the fine-tuned model weights
model.eval()  # Set the model to evaluation mode

# Send the image to the same device as the model
transformed_image = transformed_image.to(device)
model = model.to(device)

# Make a prediction
with torch.no_grad():
    outputs = model(transformed_image)
    _, predicted = torch.max(outputs, 1)


predicted_class = classes[predicted.item()]
print(f"The image is classified as class: {predicted_class}")