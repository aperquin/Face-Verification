from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

input_img = "/dataset/frame_images_DB/Alicia_Witt/5/5.2247.jpg"
output_img = "/output/Alicia_Witt_5_2247"
image_size = 160
margin = 32

mtcnn = MTCNN(image_size=image_size, margin=margin)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open(input_img)
img_cropped = mtcnn(img, save_path="/output/Alicia_Witt_5_2247_cropped.jpg")