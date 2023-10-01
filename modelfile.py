from keras.models import Sequential, load_model

# opening and store file in a variable

model=load_model('resnet.h5')
class_names=['dew','fogsmog','frost','glaze','hail','lightning','rain','rainbow','rime','sandstrome','snow']
from PIL import Image
from resizeimage import resizeimage
import matplotlib as plt
import  numpy as np
# img_path=input("enter path of Image")
def predicting_model(path):
    imgpt=path
    img=Image.open(imgpt)
    # plt.imshow(img)
    # image=image.resize((64,64,3))
    image = resizeimage.resize_contain(img, [256, 256])
    image = image.convert("RGB")
    image = np.array(image)
    image=image/255
    print(image.shape)
    pred_label = class_names[np.argmax(model.predict(image[np.newaxis,...]))]
    # pred_label=model.predict(image)
    print(pred_label)
    return pred_label



from distutils.log import debug
from fileinput import filename
from flask import *  
app = Flask(__name__)  
  
@app.route('/')  
def main():  
    return render_template("index.html")  
  
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)  
        print(f.filename)
        pred=predicting_model(f.filename)
        return render_template(f"Acknowledgement.html", name = pred) 
 
  
if __name__ == '__main__':  
    app.run(debug=True)