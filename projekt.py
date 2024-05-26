import cv2
import numpy as np

def matchTempFunction(example_img, template_images, method):
    print("MATCH TEMPLATE")
    values = []
    methods = [cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF]

    val = 0
    flag = False

    for i in range(len(template_images)):
        result = cv2.matchTemplate(example_img, template_images[i], method)

        if method == cv2.TM_CCOEFF or method == cv2.TM_CCORR:
            flag = True
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            print(str((i+1)*10) + " max val = ", max_val)
            values.append(max_val)
        elif method == cv2.TM_SQDIFF:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            print(str((i+1)*10) + " min val = ", min_val)
            values.append(min_val)    
            # val = min(values)        
        else:
            print("Nie ma takiej metody")

    if flag:
        val = max(values)
        index = values.index(val)
        print("Ograniczenie prędkości: " + str((index+1)*10) + " | " + "Wskaźnik max_val: " + str(val))
    else:
        val = min(values)
        index = values.index(val)
        print("Ograniczenie prędkości: " + str((index+1)*10) + " | " + "Wskaźnik min_val: " + str(val))    


    cv2.imshow("Dopasowany wzorzec", template_images[index])

def recognize_speed_limit_sign(image):
    width = image.shape[0]   
    print(width)
    length = image.shape[1]
    image = cv2.resize(image, (int(0.5*width), int(0.5*length)))
    # print(image.shape)

    cv2.imshow("obraz wejsciowy", image)
    cv2.waitKey()
    
    # Konwertuj obraz na przestrzeń barw HSV

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("przestrzen barw hsv", hsv)
    cv2.waitKey()

# Zdefiniuj zakres kolorów czerwonych dla pierścienia
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([179, 255, 255])

# Utwórz maskę, która zawiera tylko piksele w zakresie kolorów czerwonych
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    print(mask2.shape)
    # połączenie masek w jedną
    mask = cv2.bitwise_or(mask1, mask2)
    print("mask = ", mask.shape)
    cv2.imshow("mask1", mask1)
    cv2.waitKey()
    cv2.imshow("mask2", mask2)
    cv2.waitKey()
    cv2.imshow("mask", mask)
    # cv2.waitKey()
    
# Wykonaj operacje morfologiczne na masce głównie w celu usunięcia szumów i ogólnej poprawy jakości
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("otwarcie mask", mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("zamkniecie mask", mask)
    cv2.waitKey()
    print("mask shape = ", mask.shape)

# Znajdź kontury obiektów na masce
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy.shape)
    print(hierarchy)
    for i in hierarchy:
        print(i)
# Wyodrębnij największy kontur, który powinien reprezentować znak ograniczenia prędkości
# Przeiteruj przez kontury i hierarchię
    for i, contour in enumerate(contours):
        # Sprawdź hierarchię
        if hierarchy[0][i][3] != -1:  # Sprawdź czy kontur ma rodzica (czy jest wewnętrzny)
            # Wyświetl wewnętrzne kontury
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
            cv2.imshow("Image", image)
            # Stwórz maskę dla obszaru wewnątrz konturu pierścienia
            mask_internal = np.zeros_like(mask)
            # cv2.imshow("mask internal", mask_internal)
            cv2.drawContours(mask_internal, [contour], 0, 255, -1)
            cv2.imshow("mask internal", mask_internal) 
            # Zastosuj maskę na obrazie wejściowym
            inner_area = cv2.bitwise_and(image, image,  mask=mask_internal)
            # print(inner_area)
            # Wyświetl obszar wewnątrz pierścienia
            cv2.imshow("Inner Area", inner_area)
            cv2.waitKey()
            # cv2.destroyAllWindows()

    inner_area = cv2.cvtColor(inner_area, cv2.COLOR_BGR2GRAY)   
    cv2.imshow("Gray inner area", inner_area)      
    cv2.waitKey()
    _, binarized_inner_area = cv2.threshold(inner_area, 80, 255, cv2.THRESH_BINARY)
    print("binarized_inner_area = ", binarized_inner_area.shape)              
      
    contours, hierarchy = cv2.findContours(binarized_inner_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("binarized inner area", binarized_inner_area)
    cv2.waitKey()
    cv2.destroyAllWindows()
    largest_contour = max(contours, key=cv2.contourArea)


# Narysuj kontur na obrazie

    # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Wyodrębnij prostokąt otaczający największy kontur
    x, y, w, h = cv2.boundingRect(largest_contour)
    
# Wytnij obszar wewnątrz znaku ograniczenia prędkości
    speed_limit = binarized_inner_area[y:y+h, x:x+w]
    print(speed_limit.shape)

    cv2.imshow("speed_limit", speed_limit)
    cv2.waitKey()

# sprowadzenie rozmiaru przetworzonego obrazka do rozmiaru obrazków wzorcowych i zastosowanie
# filtru gaussowskiego do wygładzenie 
    speed_limit = cv2.resize(speed_limit, (202, 177))
    speed_limit = cv2.GaussianBlur(speed_limit, (5, 5), 0)

    speed_limit = find_min_rectangles(speed_limit)
    cv2.waitKey()

    return speed_limit


def find_min_rectangles(image):
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = []

    # Dla każdego konturu oblicz minimalny prostokąt otaczający
    for contour in contours:
        # Oblicz minimalny prostokąt otaczający
        rect = cv2.minAreaRect(contour)
        bounding_rects.append(rect)
        # Pobierz współrzędne wierzchołków prostokąta
        box = cv2.boxPoints(rect)
        # Konwertuj współrzędne na typ int
        box = np.int0(box)
        # Narysuj prostokąt na obrazie
        image = cv2.drawContours(image, [box], 0, (255, 0, 0), 2)

    return image


if __name__ == '__main__':

    img10 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/10.png", 0)
    img20 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/20.png", 0)
    img30 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/30.png", 0)
    img40 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/40.png", 0)
    img50 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/50.png", 0)
    img60 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/60.png", 0)
    img70 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/70.png", 0)
    img80 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/80.png", 0)
    img90 = cv2.imread("C:/Users/Radek/Desktop/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia/Wzorce/90.png", 0)

    template_images = [img10, img20, img30, img40, img50, img60, img70, img80, img90]
    
    image = cv2.imread("C:/Users/Radek/Desktop/Pulpit/SEMESTR 6/WM/projekt_rozpoznawanie_ograniczen_predkosci/Znaki/Zdjecia /343700885_796300525210972_6589658283176815696_n.jpg")
    speed_limit_sign = recognize_speed_limit_sign(image)

    # czasami cyfry w obrazie po przetworzeniu częściowo zanikają, można wtedy dla danego przypadku zwiększyć dolny próg przy binaryzacji
    # dla TM_SQDIFF wykrywa 20
    # dla TM_SQDIFF wykrywa 30
    # dla TM_CCOEFF wykrywa 40,   
    # dla TM_SQDIFF wykrywa 50, 5 troche zanika, ale można zwiększyć dolny próg przy binaryzacji obszaru wewnątrz znaku z 80 na np. 100 
    # dla TM_SQDIFF wykrywa 60
    # dla TM_SQDIFF wykrywa 70
    # dla TM_CCOEFF wykrywa 80 
    matchTempFunction(speed_limit_sign, template_images, cv2.TM_SQDIFF)

    print(speed_limit_sign.shape) 
    cv2.imshow("obrazek po przetworzeniu", speed_limit_sign)
    cv2.waitKey()
