from django.shortcuts import render

# Create your views here.

from django.http import HttpResponseRedirect
from django.urls import reverse

from dam_check.models import PixUpload
from dam_check.forms import ImageForm

def index(request):
    return render(request, 'index.html')

def toolkit(request):
    image_path = ""
    image_path1 = ""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PixUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()
            
            return HttpResponseRedirect(reverse('toolkit'))

    else:
        form = ImageForm()

    documents = PixUpload.objects.all()
    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/' + image_path

        document.delete()

    request.session['image_path'] = image_path
    return render(request, 'toolkit.html',
    {'documents': documents, 'image_path1': image_path1, 'form': form})

# Coding the third page of the website
import os
import json
import h5py
import numpy as np
import pickle as pk
from PIL import Image

from keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############### Prepare Image for Processsing ##############

def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

with open('static/cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)


# Shortlisting 27 categories from VGG16 for Cars/Vehicles
cat_list = [k for k, v in cat_counter.most_common()[:27]]

global graph
graph = tf.compat.v1.get_default_graph()
# this is protect agains memeroy leak

def prepare_flat(img_224):
    base_model = load_model('static/vgg16.h5')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat

CLASS_INDEX_PATH = 'static/imagenet_class_index.json'

def get_predictions(preds, top=5):
    global CLASS_INDEX
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []

    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

# Cheking if loaded image is that of a car
def car_categories_check(img_224):
    first_check = load_model('static/vgg16.h5')
    print('Validating that the loaded image is a car....')
    out = first_check.predict(img_224)
    top = get_predictions(out, top=5)

    for j in top[0]:
        if j[0:2] in cat_list:
            print('Car check passed.\n')
            return True
    return False

# Checking if car is damaged
def car_damage_check(img_flat):
    second_check = pk.load(open('static/second_check.pickle', 'rb'))
    print('Verifying that damage exist on car.....')
    preds = second_check.predict(img_flat)
    dam_labels = ['00-damage', '01-whole']
    prediction = dam_labels[preds[0]]

    if prediction == '00-damage':
        print('\tValidation complete - proceeding to damage location and severity determination.\n')
        print('\n')
        return True
    else:
        return False

def location_assessment(img_flat):
    third_check = pk.load(open('static/third_check.pickle', 'rb'))
    print('Validating the location of the damage - Front, Rea or Side.....')
    preds = third_check.predict(img_flat)
    loc_labels = ['Front', 'Rear', 'Side']
    prediction = loc_labels[preds[0]]

    print('\tThe car is damaged by the - ' + prediction.upper())
    print('\tCar damage assessment complete.\n')
    print('\n')

    return prediction   

def severity_assessment(img_flat):
    fourth_check = pk.load(open('static/fourth_check.pickle', 'rb'))
    print('Validating the damage severity...')
    preds = fourth_check.predict(img_flat)
    sev_label = ['Minor', 'Moderate', 'Severe']
    prediction = sev_label[preds[0]]

    print('\tCar damage impact is - ', prediction.upper())
    print('\tSeverity assessment complete.\n')

    print('Thank you for using this car damage assessment kit.')
    print('Copyright johnsonojeniyi@gmail.com (All right reserved)\n')
    print('\n')

    return prediction

def engine(request):
    myCar = request.session['image_path']
    img_path = myCar
    request.session.pop('image_path', None)
    request.session.modified = True

    with graph.as_default():
        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        g1 = car_categories_check(img_224)
        g2 = car_damage_check(img_flat)

        while True:
            try:
                if g1 is False:
                    g1_pix = '\tAre you sure the image loaded is that of a car? Please check and reload clearer picture of the damaged car.'
                    g2_pix = 'N/A'
                    g3 = 'N/A'
                    g4 = 'N/A'
                    ns = 'N/A'
                    break
                else:
                    g1_pix = "\tConfirmed. Loaded image is a car"

                if g2 is False:
                    g2_pix = '\tAre you sure your car is damaged?'
                    g3 = 'N/A'
                    g4 = 'N/A'
                    ns = 'N/A'
                    break
                else:
                    g2_pix = '\tCar damaged. Refer to section below for location and severity of damage.'
                    g3 = location_assessment(img_flat)
                    g4 = severity_assessment(img_flat)
                    ns = "a). Create a report and send to vendor \n b). Proceed to cost estimation \nc). Estimate TAT"  
                    break

            except:
                break

    src = 'pix_upload/'

    for img in os.listdir(src):
        if img.endswith('.jpg'):
            os.remove(src + img)

    K.clear_session()

    return render(request, 'results.html',
    context={'g1_pix': g1_pix, 'g2_pix': g2_pix, 'loc': g3, 'sev': g4, 'ns': ns})
