import sys
import numpy as np
from PIL import Image
from scipy import fftpack
import huffmanEncode
from bitstream import BitStream


zigzagOrder = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
                           49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

std_luminance_quant_tbl = np.array(
    [16,  11,  10,  16,  24,  40,  51,  61,
     12,  12,  14,  19,  26,  58,  60,  55,
     14,  13,  16,  24,  40,  57,  69,  56,
     14,  17,  22,  29,  51,  87,  80,  62,
     18,  22,  37,  56,  68, 109, 103,  77,
     24,  35,  55,  64,  81, 104, 113,  92,
     49,  64,  78,  87, 103, 121, 120, 101,
     72,  92,  95,  98, 112, 100, 103,  99], dtype=int)
std_luminance_quant_tbl = std_luminance_quant_tbl.reshape([8, 8])

std_chrominance_quant_tbl = np.array(
    [17,  18,  24,  47,  99,  99,  99,  99,
     18,  21,  26,  66,  99,  99,  99,  99,
     24,  26,  56,  99,  99,  99,  99,  99,
     47,  66,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99], dtype=int)
std_chrominance_quant_tbl = std_chrominance_quant_tbl.reshape([8, 8])


def quant_tbl_quality_scale(quality):
    if (quality <= 0):
        quality = 1
    if (quality > 100):
        quality = 100
    if (quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
    luminanceQuantTbl = np.array(np.floor(
        (std_luminance_quant_tbl * qualityScale + 50) / 100))
    luminanceQuantTbl[luminanceQuantTbl == 0] = 1
    luminanceQuantTbl[luminanceQuantTbl > 255] = 255
    luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8]).astype(int)

    chrominanceQuantTbl = np.array(np.floor(
        (std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl[chrominanceQuantTbl > 255] = 255
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)
    return luminanceQuantTbl, chrominanceQuantTbl

def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = 16 +  65.481/255 * r + 128.553/255 * g + 24.966/255 * b
    # Cb
    cbcr[:,:,1] = 128 - 37.797/255 * r - 74.203/255 * g + 112.0/255 * b
    # Cr
    cbcr[:,:,2] =  128 +  112.0/255 * r - 93.786/255 * g - 18.214/255 * b
    return np.uint8(cbcr)

def dct2d(block):
    N = block.shape[0]
    result = np.zeros_like(block, dtype=np.float32)
    for u in range(N):
        for v in range(N):
            sum_val = 0.0
            for x in range(N):
                for y in range(N):
                    sum_val += block[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            alpha_u = np.sqrt(1/N) if u == 0 else np.sqrt(2/N)
            alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
            result[u, v] = alpha_u * alpha_v * sum_val
    return result

def read_input_img(image):
    withImg, heightImg = image.size
    imgMatrix = np.array(image)
    return withImg, heightImg, imgMatrix

def compression_img(inputFile, quality):
    # set up quantization table
    luminanceQuantTbl, chrominanceQuantTbl = quant_tbl_quality_scale(quality)

    # read input image and get with, high, data of .ppm image ogiginal
    withImg, heightImg, imgMatrix = read_input_img(inputFile)

    # add matrix [0, 0, 0...] if with, height of the image is not a multiple of 8
    withImg_ = withImg
    heightImg_ = heightImg

    if (withImg_ % 8 != 0):
        withImg_ = withImg_ // 8 * 8 + 8
    if (heightImg_ % 8 != 0):
        heightImg_ = heightImg_ // 8 * 8 + 8

    print('new size of the image(maybe similar to original size)',
          withImg_, 'x', heightImg_)

    # create a new matrix image and copy the value
    newImgMatrix = imgMatrix.copy()
    
    # convert RGB to YCrCb
    newImgMatrix = rgb2ycbcr(newImgMatrix)

    # get channel Y, Cb, Cr
    Y_matrix = (newImgMatrix[:, :, 0] - 128).astype(np.int8)
    Cb_matrix = (newImgMatrix[:, :, 1] - 128).astype(np.int8)
    Cr_matrix = (newImgMatrix[:, :, 2] - 128).astype(np.int8)

    # divide block 8x8
    totalBlock = int((withImg_ / 8) * (heightImg_ / 8))
    currentBlock = 0

    Y_DC = np.zeros([totalBlock], dtype=int)
    Cb_DC = np.zeros([totalBlock], dtype=int)
    Cr_DC = np.zeros([totalBlock], dtype=int)
    d_Y_DC = np.zeros([totalBlock], dtype=int)
    d_Cb_DC = np.zeros([totalBlock], dtype=int)
    d_Cr_DC = np.zeros([totalBlock], dtype=int)

    sosBitStream = BitStream()
    for i in range(0, heightImg_, 8):
        for j in range(0, withImg_, 8):
            
            # DCT
            Y_DCTMatrix = fftpack.dct(fftpack.dct(
                Y_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
            Cb_DCTMatrix = fftpack.dct(fftpack.dct(
                Cb_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
            Cr_DCTMatrix = fftpack.dct(fftpack.dct(
                Cr_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T

            # quantization
            Y_QuantMatrix = np.rint(
                Y_DCTMatrix / luminanceQuantTbl).astype(int)
            Cb_QuantMatrix = np.rint(
                Cb_DCTMatrix / chrominanceQuantTbl).astype(int)
            Cr_QuantMatrix = np.rint(
                Cr_DCTMatrix / chrominanceQuantTbl).astype(int)

            # run length
            Y_ZZcode = Y_QuantMatrix.reshape([64])[zigzagOrder]
            Cb_ZZcode = Cb_QuantMatrix.reshape([64])[zigzagOrder]
            Cr_ZZcode = Cr_QuantMatrix.reshape([64])[zigzagOrder]

            Y_DC[currentBlock] = Y_ZZcode[0]
            Cb_DC[currentBlock] = Cb_ZZcode[0]
            Cr_DC[currentBlock] = Cr_ZZcode[0]

            if (currentBlock == 0):
                d_Y_DC[currentBlock] = Y_DC[currentBlock]
                d_Cb_DC[currentBlock] = Cb_DC[currentBlock]
                d_Cr_DC[currentBlock] = Cr_DC[currentBlock]
            else:
                d_Y_DC[currentBlock] = Y_DC[currentBlock] - Y_DC[currentBlock-1]
                d_Cb_DC[currentBlock] = Cb_DC[currentBlock] - Cb_DC[currentBlock-1]
                d_Cr_DC[currentBlock] = Cr_DC[currentBlock] - Cr_DC[currentBlock-1]
            
            if (currentBlock == 0):
                print('std_luminance_quant_tbl:\n', std_luminance_quant_tbl)
                print('block8x8:\n', Y_matrix[i:i + 8, j:j + 8])
                print('DCT:\n', Y_DCTMatrix)
                print('luminanceQuantTbl:\n', luminanceQuantTbl)
                print('quantizated block8x8:\n', Y_QuantMatrix)
                print('zigzag:\n', Y_ZZcode)
          
            # encode y_DC
            sosBitStream.write(huffmanEncode.encodeDCToBoolList(
                d_Y_DC[currentBlock], 1, 1, currentBlock), bool)

            # encode y_AC

            huffmanEncode.encodeACBlock(
                sosBitStream, Y_ZZcode[1:], 1, 1, currentBlock)

            # encode Cb_DC

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(
                d_Cb_DC[currentBlock], 0), bool)
            # encode Cb_AC

            huffmanEncode.encodeACBlock(
                sosBitStream, Cb_ZZcode[1:], 0)

            # encode Cr_DC

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(
                d_Cr_DC[currentBlock], 0), bool)
            # encode Cr_AC

            huffmanEncode.encodeACBlock(
                sosBitStream, Cr_ZZcode[1:], 0)

            currentBlock = currentBlock + 1
           
          
    # create and open output file
    jpegFile = open('output_dog.jpg', 'wb+')

    # write jpeg header
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))

    # write Y Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    luminanceQuantTbl = luminanceQuantTbl.reshape([64])
    jpegFile.write(bytes(luminanceQuantTbl.tolist()))

    # write u/v Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([64])
    jpegFile.write(bytes(chrominanceQuantTbl.tolist()))

    # write height and width
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(heightImg)[2:]
    while len(hHex) != 4:
        hHex = '0' + hHex

    jpegFile.write(huffmanEncode.hexToBytes(hHex))

    wHex = hex(withImg)[2:]
    while len(wHex) != 4:
        wHex = '0' + wHex

    jpegFile.write(huffmanEncode.hexToBytes(wHex))

    # 03    01 11 00    02 11 01    03 11 01
    # 1：1	01 11 00	02 11 01	03 11 01
    # 1：2	01 21 00	02 11 01	03 11 01
    # 1：4	01 22 00	02 11 01	03 11 01

    # write Subsamp
    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    # write huffman table
    jpegFile.write(huffmanEncode.hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))

    # SOS Start of Scan
    # yDC yAC uDC uAC vDC vAC
    sosLength = sosBitStream.__len__()
    filledNum = 8 - sosLength % 8
    if (filledNum != 0):
        sosBitStream.write(np.ones([filledNum]).tolist(), bool)

    # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00
    jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0]))

    # write encoded data
    sosBytes = sosBitStream.read(bytes)
    for i in range(len(sosBytes)):
        jpegFile.write(bytes([sosBytes[i]]))
        if (sosBytes[i] == 255):
            jpegFile.write(bytes([0]))  # FF to FF 00

    # write end symbol
    jpegFile.write(bytes([255, 217]))  # FF D9
    jpegFile.close()

    

