import math
import cv2
import matplotlib.pyplot as plt


learning_rate = 0.1
INPUT_IMAGE_SRC = "C:/Users/sukum/OneDrive/Academic/Grad_Rutgers/Intro To AI (520)/Assignments/Assignment_4/image_colorizer/ip1.jpg"
weight_mat = []
weight_mat.append([[1 for x in range(9+1)] for y in range(6)])
weight_mat.append([[1 for x in range(6+1)] for y in range(3)])

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
    return [b, g, r]

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
            error_list = [x - ip_img[row][col][i] for i, x in enumerate(predicted_val)]
            total_error = []

            weight_sum = [[] for x in range(len(weight_mat))]

            for i in range(len(weight_mat) - 1, -1, -1):
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
                        zip(new_error_list, )

                error_list = new_error_list



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

    print(loss)
