import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
import random


load = lambda path: [
    cv.imread(os.path.join(path, filename)) for filename in os.listdir("train")
]

train_images = load("train")

test_image_path = random.choice(os.listdir("test"))
print(f"Test image: {test_image_path}")
test_image = cv.imread(os.path.join("test", test_image_path))


sift = cv.SIFT_create()
bf = cv.BFMatcher()

test_keypoints, test_descriptors = sift.detectAndCompute(test_image, None)

scores = []
percent = 0

for image in train_images:
    keypoints, descriptors = sift.detectAndCompute(image, None)

    matches = bf.knnMatch(test_descriptors, descriptors, 2)

    percent += 1
    print(f"Loading... {int(percent / len(train_images) * 100)}%")

    good_matches = []
    for a, b in matches:
        if a.distance < 0.75 * b.distance:
            good_matches.append(a)
    scores.append(len(good_matches))

percent = 0

scores = np.array(scores)

max_score = np.max(scores)
max_index = np.argmax(scores)

best_image = train_images[max_index]

best_image_keypoints, best_image_descriptors = sift.detectAndCompute(best_image, None)
best_match = bf.match(test_descriptors, best_image_descriptors)


result = cv.drawMatches(
    test_image,
    test_keypoints,
    best_image,
    best_image_keypoints,
    best_match[:10],
    None,
    flags=2,
)

cv.putText(
    result,
    f"Score: {max_score}",
    (10, 130),
    cv.FONT_HERSHEY_SIMPLEX,
    5,
    (255, 0, 0),
    2,
    cv.LINE_AA,
)

plt.imshow(result)
plt.show()
