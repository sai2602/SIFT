import cv2 as cv

THRESHOLD_DISTANCE = 600  # Benchmarked with trial and error on BIPED dataset

image_1 = cv.imread('C:\\Users\\SaiParimi\\Desktop\\Dexined\\Dexined'
                    '\\Data\\BIPED\\BIPED\\edges\\imgs\\train\\rgbr\\real\\RGB_046.jpg', cv.IMREAD_GRAYSCALE)

image_2 = cv.imread('C:\\Users\\SaiParimi\\Desktop\\Dexined\\Dexined'
                    '\\Data\\BIPED\\BIPED\\edges\\imgs\\train\\rgbr\\real\\RGB_047.jpg', cv.IMREAD_GRAYSCALE)

sift = cv.xfeatures2d.SIFT_create()

keypoint_1, descriptor_1 = sift.detectAndCompute(image_1, None)
keypoint_2, descriptor_2 = sift.detectAndCompute(image_2, None)

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches = bf.match(descriptor_1, descriptor_2)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:1000]

distance_averaged = 0.0

for each_distance in matches:
    distance_averaged += each_distance.distance

distance_averaged /= 1000.0

if distance_averaged <= THRESHOLD_DISTANCE:
    print("Images are similar")
    matched_img = cv.drawMatches(image_1, keypoint_1, image_2, keypoint_2, matches, image_2, flags=2)
    cv.imshow('image', matched_img)
    cv.waitKey(0)
else:
    print("Images are not similar")
