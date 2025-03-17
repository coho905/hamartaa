from torchvision import transforms
from PIL import Image

# Define the transformation pipeline to just resize the image
transform = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
])

# Open the image
img = Image.open('photos/german_shep.jpg')

# Apply the transformations (resize in this case)
transformed_image = transform(img)

# Save the transformed image
transformed_image.save('photos/german_shep.jpg')
