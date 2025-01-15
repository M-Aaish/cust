import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import webcolors

# Function to calculate the center of a contour
def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

# Function to check overlap between triangles
def is_overlapping(center, existing_centers):
    max_overlap_dist = 20
    for ec in existing_centers:
        if np.linalg.norm(np.array(center) - np.array(ec)) < max_overlap_dist:
            return True
    return False

# Function to handle dynamic categories with angles, tolerances, and RGB selection
def handle_category(index):
    st.subheader(f"Category {index + 1}")
    
    # Angles and tolerances in one line using columns
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    angles = []
    
    with col1:
        angle_1 = st.number_input(f"Angle 1 (°)", value=60.0, step=0.1, min_value=0.0, key=f'angle_{index}_1')
    with col2:
        tolerance_1 = st.number_input(f"Tolerance 1 (°)", value=1.0, step=0.1, key=f'tolerance_{index}_1')
        
    with col3:
        angle_2 = st.number_input(f"Angle 2 (°)", value=60.0, step=0.1, min_value=0.0, key=f'angle_{index}_2')
    with col4:
        tolerance_2 = st.number_input(f"Tolerance 2 (°)", value=1.0, step=0.1, key=f'tolerance_{index}_2')
        
    with col5:
        angle_3 = st.number_input(f"Angle 3 (°)", value=60.0, step=0.1, min_value=0.0, key=f'angle_{index}_3')
    with col6:
        tolerance_3 = st.number_input(f"Tolerance 3 (°)", value=1.0, step=0.1, key=f'tolerance_{index}_3')

    # Save the angles and tolerances into a list of tuples
    angles = [(angle_1, tolerance_1), (angle_2, tolerance_2), (angle_3, tolerance_3)]
    
    # RGB Color Selection
    with col7:
        hex_color = st.color_picker(f"Select RGB Color for Category {index + 1}", key=f'rgb_color_{index}')
    
    # Convert hex to RGB
    try:
        rgb_color = webcolors.hex_to_rgb(hex_color)
    except ValueError:
        rgb_color = (255, 0, 0)  # Default to red if there's an issue with the hex color

    # Checkbox for strict matching (all angles or any angle)
    match_all_angles = st.checkbox(f"Match all angles?", value=True, key=f"checkbox_{index}")

    # Check if the sum of angles is equal to 180
    if sum([angle for angle, _ in angles]) != 180:
        st.warning("The sum of the angles must be exactly 180°. Please adjust the angles.")
    
    return angles, rgb_color, match_all_angles

# Function to process the uploaded image and detect triangles
def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale and apply binary threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Erosion and dilation
    kernel_size = (3, 3)
    kernel = np.ones(kernel_size, np.uint8)

    eroded = cv2.erode(th, kernel, iterations=3)
    dilated = cv2.dilate(th, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Copy image for drawing results
    image_copy = image.copy()

    # Set thresholds for filtering
    min_area = 500  # Minimum area of triangles to draw
    drawn_centers = []

    # Process contours from eroded image
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3 and cv2.contourArea(approx) > min_area:
            center = get_center(approx)
            if center and not is_overlapping(center, drawn_centers):
                drawn_centers.append(center)
                cv2.drawContours(image_copy, [approx], -1, (0, 0, 255), 3)  # Default red for triangles

    # Process contours from thresholded image
    for contour in contours1:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3 and cv2.contourArea(approx) > min_area:
            center = get_center(approx)
            if center and not is_overlapping(center, drawn_centers):
                drawn_centers.append(center)
                cv2.drawContours(image_copy, [approx], -1, (0, 0, 255), 3)  # Default red for triangles

    return image_copy, contours, contours1

# Main function to organize the layout
def main():
    st.title("Image Triangle Detection and Category Management")

    # Image upload
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load and process the uploaded image
        image = Image.open(uploaded_file)
        img_path = f"temp_image.jpg"
        image.save(img_path)

        # Process the image and get contours and the output image with triangles
        image_with_triangles, contours, contours1 = process_image(img_path)

        # Show original and processed images side by side
        
        # Category management section
        if 'categories' not in st.session_state:
            st.session_state.categories = []  # Initialize categories as an empty list

        # Create or delete categories
        if st.button("Add Category"):
            st.session_state.categories.append({
                'angles': [],
                'rgb_color': (255, 0, 0),  # Default color red
                'match_all_angles': True
            })

        # Delete Category
        category_to_delete = st.selectbox("Select Category to Delete", [""] + [f"Category {i+1}" for i in range(len(st.session_state.categories))], key="category_to_delete")
        
        if category_to_delete:
            category_index = int(category_to_delete.split()[-1]) - 1
            if st.button(f"Delete {category_to_delete}"):
                del st.session_state.categories[category_index]
                st.session_state.categories = st.session_state.categories  # Refresh the list after deletion
        
        # Display and handle each category
        for index, category in enumerate(st.session_state.categories):
            angles, rgb_color, match_all_angles = handle_category(index)
            st.session_state.categories[index]['angles'] = angles
            st.session_state.categories[index]['rgb_color'] = rgb_color
            st.session_state.categories[index]['match_all_angles'] = match_all_angles

            # Apply color to triangles based on category
            for contour in contours:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 3 and cv2.contourArea(approx) > 500:
                    triangle_angles = [60.0, 60.0, 60.0]  # Dummy values for angles, calculate using actual method
                    angle_match = False
                    if match_all_angles:
                        if all(angle - tolerance <= triangle_angle <= angle + tolerance for triangle_angle, (angle, tolerance) in zip(triangle_angles, angles)):
                            angle_match = True
                    else:
                        if any(angle - tolerance <= triangle_angle <= angle + tolerance for triangle_angle, (angle, tolerance) in zip(triangle_angles, angles)):
                            angle_match = True
                    if angle_match:
                        cv2.drawContours(image_with_triangles, [approx], -1, rgb_color, 3)

        # Provide download functionality for the processed image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(image_with_triangles, caption="Detected Triangles", use_column_width=True)
        image_with_triangles= cv2.cvtColor(image_with_triangles, cv2
                                           .COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', image_with_triangles)
        img_bytes = buffer.tobytes()
        
        
        st.download_button( 
            label="Download Processed Image",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    import os
    import sys

    if getattr(sys, 'frozen', False):  # If running as a bundled executable
        os.system("streamlit run " + sys.executable)
    else:  # If running normally as a Python script
        import streamlit as st
        # Your Streamlit app code here

    main()
