import math
import random as rnd
import cv2
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.1
filename = "ip1_min.jpg"
INPUT_IMAGE_SRC = "C:/Users/sukum/OneDrive/Academic/Grad_Rutgers/Intro To AI (520)/Assignments/Assignment_4/image_colorizer/" + filename
weight_mat = []
weight_mat.append([[rnd.uniform(0, 1) for x in range(6)] for y in range(9+1)])
weight_mat.append([[rnd.uniform(0, 1) for x in range(3)] for y in range(6+1)])

error_list = []

def generate_grid(i, j, img):
    height, width = img.shape
    ans = []

    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            if x < 0 or y < 0 or x >= height or y >= width:
                ans.append(0)
            else:
                ans.append(img[x][y])
    return ans

def predict(input_values):
    r, g, b, = 10, 10, 10
    # curr_inputs = np.matrix(input_values + [1], dtype=float)
    curr_inputs = input_values
    for i in range(len(weight_mat)):
        # curr_weights = np.matrix(weight_mat[i], dtype=float)
        # curr_inputs = np.matmul(curr_inputs, curr_weights)
        curr_inputs = np.matmul(curr_inputs + [1], weight_mat[i])
        max_val = max(abs(curr_inputs))
        curr_inputs = list(map(lambda x: x/max_val, curr_inputs))
        curr_inputs = list(map(lambda x: sigmoid(x), curr_inputs))

    return curr_inputs

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    ip_img = cv2.imread(INPUT_IMAGE_SRC)
    # cv2.imshow("Input", ip_img)
    ip_blue, ip_green, ip_red = cv2.split(ip_img)


    ip_gray = cv2.cvtColor(ip_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Input", ip_gray)

    ip_gray_height, ip_gray_width = ip_gray.shape
    loss = None

    # Traverse each pixel
    for row in range(ip_gray_height):
        for col in range(ip_gray_width):
            # for channel in range(3):
            input_values = generate_grid(row, col, ip_gray)
            predicted_val = predict(input_values)
            # TODO: seperate each channel into independent neural network
            # error_list = [m+n for m, n in zip(error_list, list(map(lambda z: z**2, [x - ip_img[row][col][i] for i, x in enumerate(predicted_val)])))]
            error_list = [ip_img[row][col][i] - (x*255) for i, x in enumerate(predicted_val)]
            epoc_loss = []

            weight_sum = [[] for x in range(len(weight_mat))]

            for i in range(len(weight_mat) - 1, -1, -1):
                weight_sum[i] = [0 for x in range(len(weight_mat[i][0]))]
                for y in range(len(weight_mat[i][0])):
                    weight_sum[i][y] = 0
                    for x in range(len(weight_mat[i])):
                        weight_sum[i][y] = weight_sum[i][y] + weight_mat[i][x][y]

            for i in range(len(weight_mat) - 1, -1, -1):
                new_error_list = [0 for x in range(len(weight_mat[i]))]
                for x in range(len(weight_mat[i])):
                    for y in range(len(weight_mat[i][0])):
                        weight_mat[i][x][y] = weight_mat[i][x][y] + learning_rate * error_list[y] * (weight_mat[i][x][y] / weight_sum[i][y])
                        new_error_list[x] = new_error_list[x] + error_list[y] * (weight_mat[i][x][y] / weight_sum[i][y])

                if i == len(weight_mat) - 1:
                    epoc_loss = error_list

                error_list = new_error_list


                print(weight_mat[0])
                print(row/ip_gray_height * 100, "%")

    # Test the Neural Network
    output_image = np.zeros((ip_img.shape[0], ip_img.shape[1], 3), np.float64)

    for row in range(ip_gray_height):
        for col in range(ip_gray_width):
            output_image[row][col] = tuple(map(lambda x: x*255, predict(generate_grid(row, col, ip_gray))))

    cv2.imwrite('op_img.jpg', output_image)
    print("done")

            # for i in range(len(weight_mat) - 1, -1, -1):
            #     for x in range(len(weight_mat[i])):
            #         for y in range(len(weight_mat[i][0])):
            #
            #             if i == len(weight_mat) - 1:
            #                 # Dealing with weights between last hidden layer and output layer
            #                 weight_mat[i][x][y] = weight_mat[i][x][y] - learning_rate * (ip_img[row][col][y] - predicted_val[y]) * predicted_val[y] * (1 - predicted_val[y])
            #             else:
            #                 pass
            #         fhsbd fdfk


