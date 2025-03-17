import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from labels import get_label

IMG_PATH = 'photos/german_shep.jpg'
animal = 'german_shep'
true_label = 235


model = models.mobilenet_v2(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

img = Image.open(IMG_PATH)
transformed_image = transform(img)
loss_function = torch.nn.CrossEntropyLoss()
def affect_image_adversarially(image, epochs, loss_function, true_label, scaling_factor):
    new_image = image.clone().detach().requires_grad_(True)
    for i in range(0, epochs):
        output = model(new_image.unsqueeze(0))
        loss = loss_function(output, torch.tensor([true_label]))
        loss.backward()
        grad = new_image.grad
        if grad is None:
            delta = 0
        else:
            delta = torch.sign(grad) * scaling_factor
        new_image = new_image.detach() + delta
        new_image = torch.clamp(new_image, 0, 1).requires_grad_(True)
        if new_image.grad is not None:
            new_image.grad.zero_()
    return new_image


adversarial_image = affect_image_adversarially(transformed_image, 150, loss_function, true_label, 0.01)
final_img = transforms.ToPILImage()(adversarial_image.squeeze(0))

with torch.no_grad():
    final_output = model(adversarial_image.unsqueeze(0))
    final_pred = torch.argmax(final_output, dim=1).item()
    softmax_output = torch.softmax(final_output, dim=1)
    final_pred_idx = int(final_pred)
    final_conf = torch.softmax(final_output, dim=1)[0][final_pred_idx].item()

print(f"Final prediction: Class {final_pred} with {final_conf:.4f} confidence")
print(f"Attack {'successful' if final_pred != true_label else 'failed'}")
print(f"True class: {get_label(true_label)}")
print(f"Predicted class: {get_label(final_pred)}")
final_img.save(f'photos/adversarial_{animal}.jpg')
final_img.show()
