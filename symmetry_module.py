import cv2
import numpy as np
from scipy.spatial import KDTree
from PIL import Image

class SymmetryAnalysis:
    def __init__(self, front_rear_img=None, side1=None, side2=None, side=0, base="left"):
        self.fr_img=front_rear_img
        self.side1_img=side1
        self.side2_img=side2
        self.side=side
        self.matching_percentage=0
        self.base=base
        self.right_points=None
        self.matched_right_points=[]
        self.unmatched_points=[]

    def build_gaussian_pyramid(self, image, levels):
        gaussian_pyramid = [image]
        for i in range(1, levels):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image)
        return gaussian_pyramid

    def build_laplacian_pyramid(self, gaussian_pyramid):
        laplacian_pyramid = []
        levels = len(gaussian_pyramid)
        for i in range(levels - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            upsampled_image = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], upsampled_image)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1])  # The smallest level
        return laplacian_pyramid

    def combine_edges(self, laplacian_pyramid, original_size):
        combined_edges = np.zeros(original_size, dtype=np.uint8)
        for laplacian in laplacian_pyramid:
            edges = cv2.Canny(laplacian, 50, 150)
            resized_edges = cv2.resize(edges, (original_size[1], original_size[0]))
            combined_edges = cv2.bitwise_or(combined_edges, resized_edges)
        return combined_edges

    def retImg(self):
        frnt_img=cv2.imread(self.fr_img)
        return frnt_img
    def calcSymmetry(self):
        if self.side==0:
            fr_img=np.array(self.fr_img)
            combined_edges=self.preprocessImg(fr_img)
            # Draw edges by plotting points (skeleton)
            points = np.column_stack(np.where(combined_edges > 0))
            skeleton_image = self.getSkeleton(points, combined_edges)
            # Split the image in half
            height, width = skeleton_image.shape
            mid_width = width // 2
            # Check for odd width and adjust if necessary
            if width % 2 != 0:
                left_half = skeleton_image[:, :mid_width]
                right_half = skeleton_image[:, mid_width+1:]
            else:
                left_half = skeleton_image[:, :mid_width]
                right_half = skeleton_image[:, mid_width:]
            # Extract points from the left half
            left_points = np.column_stack(np.where(left_half > 0))

            # Extract and mirror points from the right half
            self.right_points = np.column_stack(np.where(right_half > 0))
            self.right_points[:, 1] = mid_width-self.right_points[:, 1] - 1
            imgForGray=fr_img
        else: 
            side1_img=np.array(self.side1_img)
            side2_img=np.array(self.side2_img)
            combined_edges1=self.preprocessImg(side1_img)
            combined_edges2=self.preprocessImg(side2_img)
            # Draw edges by plotting points (skeleton)
            points1 = np.column_stack(np.where(combined_edges1 > 0))
            points2 = np.column_stack(np.where(combined_edges2 > 0))
            skeleton_image1 = self.getSkeleton(points1, combined_edges1)
            skeleton_image2 = self.getSkeleton(points2, combined_edges2)
            left_half= skeleton_image1
            right_half= skeleton_image2

            # Check for odd width and adjust if necessary
            height, width = skeleton_image1.shape
            if width % 2 != 0:
                left_half = skeleton_image1[:, :width+1]
            # else:
            #     left_half = skeleton_image1[:, :width]

            # Check for odd width and adjust if necessary
            height, width = skeleton_image2.shape
            if width % 2 != 0:
                right_half = skeleton_image2[:, :width+1]
            # else:
            #     right_half = skeleton_image2[:, :width]
            # Extract points from the left half
            left_points = np.column_stack(np.where(left_half > 0))

            # Extract and mirror points from the right half
            self.right_points = np.column_stack(np.where(right_half > 0))
            self.right_points[:, 1] = width-self.right_points[:, 1] - 1
            imgForGray=side1_img

        # Mirror the right half
        right_half_flipped = cv2.flip(right_half, 1)

        # Total points in the left half
        total_left_points = len(left_points)
        print('total points: ',total_left_points)
        matches = 0
        threshold_distance = 10  # Define a threshold distance for matching points

        # Use KDTree for efficient point matching
        left_kd_tree = KDTree(left_points)
        matched_left_indices = set()

        # To store the matched points for visualization
        self.matched_right_points = []
        self.unmatched_points=[]
        for rp in self.right_points:
            # Find the index of the nearest point in the left half within the threshold distance
            indices = left_kd_tree.query_ball_point(rp, threshold_distance)
            if indices:
                # Check if any of the matched indices have not been used before
                for index in indices:
                    if index not in matched_left_indices:
                        matches += 1
                        matched_left_indices.add(index)
                        self.matched_right_points.append(rp)
                        break
            else:
                self.unmatched_points.append(rp)

        rem_points=total_left_points-matches-len(self.unmatched_points)
        # Calculate the percentage of matching points
        if total_left_points > 0:
            self.matching_percentage = (matches / (total_left_points-rem_points)) * 100
        else:
            self.matching_percentage = 0

        # Display the number of matching points and the percentage
        print(f'Number of matching points: {matches}')
        print(f'Percentage of matching points: {self.matching_percentage:.2f}%')

        gray_image = cv2.cvtColor( imgForGray, cv2.COLOR_BGR2GRAY)
        # Overlay the mirrored right points on the left half in blue
        leftWMiRpts = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        for point in self.right_points:
            leftWMiRpts[point[0], point[1]] = [255, 0, 0]  # Red color for mirrored points
        
        # Create an overlay image to visualize the matches
        leftWRMapts = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        for point in self.matched_right_points:
            cv2.circle(leftWRMapts, (point[1], point[0]), 1, (0, 255, 0), -1)  # green color for matched points

        # Create an overlay image to visualize the matches
        leftWRUmapts = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        for point in self.unmatched_points:
            cv2.circle(leftWRUmapts, (point[1], point[0]), 1, (0, 0, 255), -1)  # Blue color for matched points
        return (self.matching_percentage, leftWMiRpts, leftWRMapts, leftWRUmapts)


    def preprocessImg(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)

        # Build Gaussian pyramid
        levels = 1  # You can adjust the number of levels
        gaussian_pyramid = self.build_gaussian_pyramid(equalized_image, levels)

        # Build Laplacian pyramid
        laplacian_pyramid = self.build_laplacian_pyramid(gaussian_pyramid)

        # Combine edges from all levels
        original_size = equalized_image.shape
        combined_edges = self.combine_edges(laplacian_pyramid, original_size)
        return combined_edges
    
    def getSkeleton(self, points, combined_edges):
        combined_edges=combined_edges
        skeleton_image = np.zeros_like(combined_edges)
        for point in points:
            skeleton_image[point[0], point[1]] = 255
        
        return skeleton_image
        
    def leftWithMirroredRight(self, img):
        # Overlay the mirrored right points on the left half in blue
        gray_image = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
        overlay_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        for point in self.right_points:
            overlay_image[point[0], point[1]] = [255, 0, 0]  # Blue color for mirrored points
        return overlay_image
    
    # def matched_right_points(self, img)