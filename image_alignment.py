import cv2
import numpy as np


MAX_FEATURES = 500
MATCH_THRESHOLD = 0.15


def alignImages(in_img, tgt_img):

  # Convert images to grayscale
  in_img_gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
  tgt_img_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute desc.
  orb = cv2.ORB_create(MAX_FEATURES)
  kp1, desc1 = orb.detectAndCompute(in_img_gray, None)
  kp2, desc2 = orb.detectAndCompute(tgt_img_gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(desc1, desc2, None)
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Delete innecesary matches
  numGoodMatches = int(len(matches) * MATCH_THRESHOLD)
  matches = matches[:numGoodMatches]

  # Draw top matches
  img_matches = cv2.drawMatches(in_img, kp1, tgt_img, kp2, matches, None)
  cv2.imwrite("matches.jpg", img_matches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  height, width, _ = tgt_img.shape
  in_imgReg = cv2.warpPerspective(in_img, h, (width, height))

  return in_imgReg, h


if __name__ == '__main__':

  # Read reference image
  refFilename = "input.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  imFilename = "target.jpg"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg, h = alignImages(im, imReference)

  # Write aligned image to disk.
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)

  # Print estimated homography
  print("Estimated homography : \n",  h)
