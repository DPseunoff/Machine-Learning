import cv2
import numpy as np


def canny(image):
    # Перевод цвета изображения в оттенки серого (grayscale)
    # В данном варианте каждый пиксель получает 1
    # канал интенсивности вместо 3, как в цветном
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Сокращение шума с помощью Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Функция, позволяющая сделать раздел в местах,
    # где идет резкий переход градиента
    canny = cv2.Canny(blur, 50, 150)
    return canny


# Определение интересующей области
# (Создание и применение маски)
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Оптимизация (получение двух усредненных линий)
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# Подсчет координат для усредненных прямых
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# Отображение построенных линий
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# Обработка изображения
def image_processing():
    # Считывание входного изображения в numpy array
    image = cv2.imread('test_image.jpg')
    #Подготовка изображения
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    # Применение преобразования Хафа
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    # Получение комбинации нашего изображения и построенных линий
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    # Вывод изображения в отдельное окно
    cv2.imshow('result', combo_image)
    cv2.waitKey(0)


# Обработка видео
def video_processing():
    cap = cv2.VideoCapture('test2.mp4')
    while cap.isOpened():
        _, frame = cap.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('result', combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print('Выберите режим:')
        print('1 - Обработка изображения')
        print('2 - Обработка видео')
        print('3 - Выход из программы')
        m = int(input())
        if m < 1 or m > 3:
            print('Неверный номер!')
            continue
        if m == 1:
            image_processing()
            continue
        if m == 2:
            video_processing()
            continue
        if m == 3:
            print('Завершение программы')
            break


main()
