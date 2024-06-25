import streamlit as st
from PIL import Image
from crop_img import RemImgBackground
import matplotlib.pyplot as plt
from symmetry_module import SymmetryAnalysis
import cv2
import mail
import numpy as np

def resizeImgs(img1, img2):
    #get the size of img 
    width1, height1=img1.size
    img1size=width1, height1
    width2, height2=img2.size
    img2size=width2, height2
    #get min of width & height 
    minwidth=min(width1, width2)
    minheight=min(height1, height2)
    #setting the equal sizes 
    minsize=(minwidth, minheight)
    # print("Original sizes:", img1size, img2size)
    # print("New size:", minsize)
    #using thumbnail for appropriate resizing
    img1.thumbnail((minsize)) 
    img2.thumbnail((minsize))
    return (img1, img2)


# Set the page configuration
st.set_page_config(page_title="Symmetry Analysis")

# Title of the app
st.title("Symmetry Analysis Project")

# Input fields for user details
user_name = st.text_input("Name of User")
user_email = st.text_input("User Email")

# Dropdown to select view type
view_type = st.selectbox("Select View Type", ["Front/Rear View", "Side View"])

# select base half 
base = st.selectbox("Select base half", ["Left", "Right"])

# Initialize session state for images and results
if 'images' not in st.session_state:
    st.session_state.images = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Conditional rendering of image upload fields based on view type
# front_rear_image = None
front_rear_image = None
side_image1 = None
side_image2 = None
side=0
rembackground=RemImgBackground()
if view_type == "Front/Rear View":
    st.write("Upload Front and Rear Images")
    front_rear_image = st.file_uploader("Upload Front/Rear Image", type=["jpg", "jpeg", "png"])
    # st.write(type(np.asarray(front_rear_image)))
    if front_rear_image:
        front_rear_image = Image.open(front_rear_image)
        front_rear_image=np.array(front_rear_image)
        front_rear_image=rembackground.remBg(front_rear_image)
        
elif view_type == "Side View":
    st.write("Upload Side Images")
    side_image1 = st.file_uploader("Upload Right Side Image", type=["jpg", "jpeg", "png"])
    side_image2 = st.file_uploader("Upload Left Side Image", type=["jpg", "jpeg", "png"])
    if side_image1 and side_image2:
        side_image1 = Image.open(side_image1)
        side_image1 = np.array(side_image1)
        side_image1 = rembackground.remBg(side_image1)
        side_image2 = Image.open(side_image2)
        side_image2 = np.array(side_image2)
        side_image2 = rembackground.remBg(side_image2)
        side=1
        side_image1, side_image2=resizeImgs(side_image1, side_image2)
        st.image(side_image1)
        st.image(side_image2)
        
images=[]
symmetry_percent=''
# Button to trigger analysis
if st.button("Perform Symmetry Analysis"):
    if front_rear_image or (side_image1 and side_image2):
        analysis = SymmetryAnalysis(front_rear_image, side_image1, side_image2, side=1 if view_type == "Side View" else 0, base=base)
        st.session_state.analysis_results = analysis.calcSymmetry()
        symmetry_percentage = round(st.session_state.analysis_results[0])
        st.subheader("Symmetry Percentage")
        st.write(f"{symmetry_percentage}%")
            
        st.session_state.images = [
            st.session_state.analysis_results[1],
            st.session_state.analysis_results[2],
            st.session_state.analysis_results[3]
        ]

# Check if analysis results are available to display the images
if st.session_state.analysis_results:
    if st.button("See the analysis"):
        st.image(st.session_state.analysis_results[1])
        st.image(st.session_state.analysis_results[2])
        st.image(st.session_state.analysis_results[3])


# Check if images are available to send via email
if st.session_state.images:
    if st.button("Get mail copy of results"):
        mail.sendEmail(user_email, st.session_state.images, user_name, symmetry_percent, view_type)

# if __name__ == "__main__":
#     st.write("Welcome to the Symmetry Analysis Project.")