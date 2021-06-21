import os
import boto3
import cv2
import pytesseract
import numpy as np
from difflib import SequenceMatcher
import time

# get grayscale image
def get_grayscale(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img

#OCR FUNCTION
def get_ocr_result(img, option):
    strOcr = pytesseract.image_to_string(get_grayscale(img), config=option).strip()
    return strOcr

def lambda_handler(event, context):

    startTime = time.time()

    s3_client = boto3.client('s3')
    
    bucket = os.environ['COMPARE_BUCKET']
    
    pdf_prifix_path = 'images/sample_102_pdf_jpg/8.jpg'
    docx_prifix_path = 'images/sample_102_docx_jpg/8.jpg'
    
    pdfImg = s3_client.get_object(Bucket=bucket, Key=pdf_prifix_path)
    docxImg = s3_client.get_object(Bucket=bucket, Key=docx_prifix_path)    

    firstLineIndex = 0
    lastLineIndex = 0

    custom_config = r'--oem 3 --psm 6 -l jpn+eng'    

    # load image
        # img_pdf = cv2.imread(pdf_image_url)
        # img_docx = cv2.imread(docx_image_url)
    pdf_content = pdfImg["Body"].read()
    np_array = np.fromstring(pdf_content, np.uint8)
    img_pdf = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    docx_content = docxImg["Body"].read()
    np_array = np.fromstring(docx_content, np.uint8)
    img_docx = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # convert to gray
    gray_pdf = cv2.cvtColor(img_pdf, cv2.COLOR_BGR2GRAY)
    gray_docx = cv2.cvtColor(img_docx, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image
    thresh_pdf = cv2.threshold(gray_pdf, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh_docx = cv2.threshold(gray_docx, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # use morphology erode to blur horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (151, 3))
    morph_pdf = cv2.morphologyEx(thresh_pdf, cv2.MORPH_DILATE, kernel)
    morph_docx = cv2.morphologyEx(thresh_docx, cv2.MORPH_DILATE, kernel)

    # use morphology open to remove thin lines from dotted lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
    morph_pdf = cv2.morphologyEx(morph_pdf, cv2.MORPH_OPEN, kernel)
    morph_docx = cv2.morphologyEx(morph_docx, cv2.MORPH_OPEN, kernel)

    # find contours
    cntrs_pdf = cv2.findContours(morph_pdf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs_pdf = cntrs_pdf[0] if len(cntrs_pdf) == 2 else cntrs_pdf[1]
    cntrs_docx = cv2.findContours(morph_docx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs_docx = cntrs_docx[0] if len(cntrs_docx) == 2 else cntrs_docx[1]

    print('1111111111111111111111111111111111111111111111')
    print('pdf line length: ', len(cntrs_pdf))
    print('docx line length: ', len(cntrs_pdf))
    print('111111111111111111111111111111111111111111111\n')

    # find the topmost box
    ythresh = 1000000
    for c in cntrs_pdf:
        box = cv2.boundingRect(c)
        x,y,w,h = box
        if y < ythresh:
            topbox = box
            ythresh = y

    # Draw contours excluding the topmost box
    result_pdf = img_pdf.copy()
    result_docx = img_docx.copy()

        
    if len(cntrs_pdf) >= 5 and len(cntrs_pdf) >= 5:
        firstLineIndex = 3
        lastLineIndex = 1
    else:
        if len(cntrs_pdf) == 4 and len(cntrs_pdf) == 4:
            firstLineIndex = 2
            lastLineIndex = 1
        else:
            if len(cntrs_pdf) == 3 and len(cntrs_pdf) == 3:
                firstLineIndex = 1
                lastLineIndex = 1
            else:
                if len(cntrs_pdf) <= 2 and len(cntrs_pdf) <= 2:
                    firstLineIndex = 0
                    lastLineIndex = 0            

    # first line images
    box_pdf_first_line = cv2.boundingRect(cntrs_pdf[len(cntrs_pdf) - firstLineIndex]) # why 3?, to avoid selecting page number
    x,y,w,h = box_pdf_first_line
    ROI = result_pdf[y:y+h, x:x+w]
    str_pdf_first_line = get_ocr_result(ROI, custom_config)
    # cv2.imshow('PDF FIRST LINE', ROI)

    box_docx_first_line = cv2.boundingRect(cntrs_docx[len(cntrs_docx) - firstLineIndex])
    x,y,w,h = box_docx_first_line
    ROI = result_pdf[y:y+h, x:x+w]
    str_docx_first_line = get_ocr_result(ROI, custom_config)
    # cv2.imshow('DOCX FIRST LINE', ROI)

    print('---------------------------  FIRST LINE ----------------------------')
    print('\n')
    print('----------------- pdf ------------------')
    print(str_pdf_first_line)
    str_pdf_first_line = str_pdf_first_line.replace(" ", "")
    print('----------------------------------------')
    print('\n')
    print('----------------- docx -----------------')
    print(str_docx_first_line)
    str_docx_first_line = str_docx_first_line.replace(" ", "")
    print('----------------------------------------')
    print('\n')
    print('-----------------percent----------------------')
    res_percent_first_line = SequenceMatcher(None, str_pdf_first_line, str_docx_first_line)
    print(round(res_percent_first_line.ratio(), 3))
    print('----------------------------------------------')
    print('\n')
    print('--------------------------------------------------------------------')


    # last line images
    box_pdf_last_line = cv2.boundingRect(cntrs_pdf[lastLineIndex])
    x,y,w,h = box_pdf_last_line
    ROI = result_pdf[y:y+h, x:x+w]
    str_pdf_last_line = get_ocr_result(ROI, custom_config)
    # cv2.imshow('PDF LAST LINE', ROI)

    box_docx_last_line = cv2.boundingRect(cntrs_docx[lastLineIndex])
    x,y,w,h = box_docx_last_line
    ROI = result_pdf[y:y+h, x:x+w]
    str_docx_last_line = get_ocr_result(ROI, custom_config)
    # cv2.imshow('DOCX LAST LINE', ROI)

    print('\n')
    print('------------------------------- LAST LINE --------------------------')
    print('\n')
    print('------------------ pdf -------------------')
    print(str_pdf_last_line)
    str_pdf_last_line = str_pdf_last_line.replace(" ", "")
    print('-------------------------------------------')
    print('\n')
    print('------------------ docx -------------------')
    print(str_docx_last_line)
    str_docx_last_line = str_docx_last_line.replace(" ", "")
    print('-------------------------------------------')
    print('\n')
    print('-----------------percent----------------------')
    res_percent_last_line = SequenceMatcher(None, str_pdf_last_line, str_docx_last_line)
    print(round(res_percent_last_line.ratio(), 3))
    print('------------------------------------------\n')
    print('--------------------------------------------------------------------')

    # cv2.waitKey(0)
    
    endTime = time.time()
    difference = int(endTime - startTime)
    print('wasted time: ', difference, ' s')

    return {
		'body': {
            'status': 200,
            'compare-result': (round(res_percent_last_line.ratio(), 3) + round(res_percent_first_line.ratio(), 3))/2
        },
		'headers': {'Content-Type': 'text/html', 'Access-Control-Allow-Origin': '*'}
	}
