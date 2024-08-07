import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import os
from django.conf import settings

# class_names = ['mri', 'no-mri']
model_path = os.path.join(settings.BASE_DIR, 'cv/image-classifier-models/image-classifier-model-2')

def get_class_of_image(img_path):
  model = keras.models.load_model(model_path)
  image = load_img(img_path, target_size=(224, 224))
  img = np.array(image)
  img = img / 255.0
  img = img.reshape(1,224,224,3)
  prob = model.predict(img)
#   clas = model.predict_proba(img)
#   print(f'{clas[0][0]}: {class_names[label[0][0]]}')
  print(prob)
  return prob[0][0]<0.5

if __name__ == '__main__':
    import os
    imgs = os.listdir('media/images')
    res = []
    for i in imgs:
        res.append([get_class_of_image_with_model('media/images/'+i), i])

    os.system('cls')
    for i in res:
        print(i)
