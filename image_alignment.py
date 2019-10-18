import cv2
import numpy as np


MAX_FEATURES = 500
MATCH_THRESHOLD = 0.01

STEP = 5


def alignImages(in_img, tgt_img):

  # Convert images to grayscale
  in_img_gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
  tgt_img_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

  h, w = in_img_gray.shape

  # Create a mask
  temp_mask = np.zeros_like(in_img_gray,dtype=np.uint8)
  temp_mask[:250, :450] = 255
  temp_mask[h-250:h, :450] = 255

  # Detect ORB features and compute desc.
  orb = cv2.ORB_create(MAX_FEATURES)
  kp1, desc1 = orb.detectAndCompute(in_img_gray, temp_mask)
  kp2, desc2 = orb.detectAndCompute(tgt_img_gray, temp_mask)

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
  h, mask = cv2.estimateAffinePartial2D(points1, points2, cv2.RANSAC)
  #h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  height, width, _ = tgt_img.shape
  in_imgReg = cv2.warpAffine(in_img, h, (width, height))
  #in_imgReg = cv2.warpPerspective(in_img, h, (width, height))

  return in_imgReg, h


def get_form_ROIs(img):
    stop = False
    pt0 = [0, 0]
    pt1 = [10, 10]
    while not stop:
        canvas = img.copy()
        cv2.rectangle(canvas, (pt0[0], pt0[1]), (pt1[0], pt1[1]), (0, 0, 255), 2)
        cv2.imshow("window", canvas)
        k = cv2.waitKey(0)
        if (k == 32 ):
            stop = True
        if (k == 52): # Numpad 4
            if pt0[0] - STEP > 0:
                pt0[0] = pt0[0] - STEP
                pt1[0] = pt1[0] - STEP
        if (k == 54): # Numpad 2
            if pt1[0] + STEP <= img.shape[0]:
                pt0[0] = pt0[0] + STEP
                pt1[0] = pt1[0] + STEP
        if (k == 56): # Numpad 6
            if pt0[1] - STEP > 0:
                pt0[1] = pt0[1] - STEP
                pt1[1] = pt1[1] - STEP
        if (k == 50): # Numpad 8
            if pt1[1] + STEP <= img.shape[1]:
                pt0[1] = pt0[1] + STEP
                pt1[1] = pt1[1] + STEP

        if (k == 97): # a
            if pt0[0] - STEP > 0:
                pt0[0] = pt0[0] - STEP
        if (k == 115): # s
            if pt1[1] + STEP <= img.shape[1]:
                pt1[1] = pt1[1] + STEP
        if (k == 100): # d
            if pt1[0] + STEP <= img.shape[0]:
                pt1[0] = pt1[0] + STEP
        if (k == 119): # w
            if pt0[1] - STEP > 0:
                pt0[1] = pt0[1] - STEP

        if (k == 65): # A
            if pt0[0] + STEP < pt1[0]:
                pt0[0] = pt0[0] + STEP
        if (k == 83): # S
            if pt1[1] - STEP > pt0[1]:
                pt1[1] = pt1[1] - STEP
        if (k == 68): # D
            if pt1[0] - STEP > pt0[0]:
                pt1[0] = pt1[0] - STEP
        if (k == 87): # w
            if pt0[1] + STEP < pt1[1]:
                pt0[1] = pt0[1] + STEP

        if (k == 13):
            print("[{}:{}, {}:{}]".format(pt0[0], pt1[0], pt0[1], pt1[1]))


        #print(k)




if __name__ == '__main__':
    '''
    img = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
    get_form_ROIs(img)
    '''


    # Read reference image
    refFilename = "Data/template-1.png"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    print(imReference.shape)

    # Read image to be aligned
    imFilename = "Data/scan_0.png"
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    print(im.shape)

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
