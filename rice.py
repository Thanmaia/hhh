
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
# image=Image.open('D:/ML/archive (3)/field.jpg')
# st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.pixelstalk.net/wp-content/uploads/2016/10/New-Nice-Wallpapers-HD.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
st.title("Dieseas Detection")

upload_file = st.file_uploader("Upload Effected Image", type = ['jpg','png','jpeg'])
generate_pred = st.button("predict")
model = tf.keras.models.load_model('C:/Users/sai manikanta/rice.h5')
def import_n_pred(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if generate_pred:
    image = Image.open(upload_file)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image, model)
    labels = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    st.title("predicted dieseas is {}".format(labels[np.argmax(pred)]))
    if labels[np.argmax(pred)]=='Leaf smut':
        st.text("<h1>Leaf smut, caused by the fungus Entyloma oryzae<\h1>, is a widely distributed\nbut somewhat minor, disease of rice. The fungus produces slightly raised, angular, black spots\n(sori) on both sides of the leaves (Figure 1). Although rare, it also can produce spots on \nleaf sheaths.Leaf smut, caused by the fungus Entyloma oryzae, is a widely distributed,\n but somewhat minor, disease of rice. The fungus produces slightly raised, angular, black spots (sori) on both \nsides of the leaves (Figure 1). Although rare, it also can produce spots on leaf sheaths.Leaf smut,\n caused by the fungus Entyloma oryzae, is a widely distributed, but somewhat minor, disease of rice.\n The fungus produces slightly raised, angular, black spots (sori) on both sides of the leaves (Figure 1).\n Although rare, it also can produce spots on leaf sheaths.")
    if labels[np.argmax(pred)] == 'Bacterial leaf blight':
        st.text("Leaf smut, caused by the fungus Entyloma oryzae, is a widely,Leaf smut,\n caused by the fungus Entyloma oryzae, is a widely,Leaf smut, caused by the fungus Entyloma oryzae,\n is a widely,Leaf smut, caused by the fungus Entyloma oryzae, is a widely")
    if labels[np.argmax(pred)] == 'Brown spot':
        st.text("Leaf smut, caused by the fungus Entyloma oryzae, is a widely,Leaf smut,\n caused by the fungus Entyloma oryzae, is a widely,Leaf smut")