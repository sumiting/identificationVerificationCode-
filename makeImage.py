import sys,shutil,random,time,os
#shutil用于文件/目录拷贝

from captcha.image import ImageCaptcha
#capcha用于生成验证码图片的库

CHAR_SET=['0','1','2','3','4','5','6','7','8','9']
#验证码字符集

CHAR_SET_LEN=10
#验证码长度，每个验证码由4个数字组成
CAPTCHA_LEN=4

CAPTCHA_IMAGE_PATH='./images/'
#验证码存放路径

TEST_IMAGE_PATH='./test/'

TEST_IMAGE_NUMBER=50
#用于模型测试的验证码图片个数，从生成的验证码图片中取出来放入测试集中

def generate_captcha_image(charSet=CHAR_SET,charSetLen=CHAR_SET_LEN,captchaImaPath=CAPTCHA_IMAGE_PATH):
    k=0
    total=1
    for i in range(CAPTCHA_LEN):
        total*=charSetLen
    for i in range(charSetLen):
        for j in range(charSetLen):
            for m in range(charSetLen):
                for n in range(charSetLen):
                    captcha_text=charSet[i]+charSet[j]+charSet[m]+charSet[n]
                    image=ImageCaptcha()#生成画布
                    image.write(captcha_text,captchaImaPath+captcha_text+'.jpg')
                    k+=1
                    sys.stdout.write("\rCreating%d%d"%(k,total))#和print类似

                    sys.stdout.flush()#每隔一秒输出一次
#从验证码的图片集中取出一部分作为测试集，这些图片不参加训练，只用于模型的检测
def prepare_test_set():
    fileNameList=[]
    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):#图片路径
        captcha_name=filePath.split('/')[-1]#通过/进行分割找到最后一个元素为图片名称
        fileNameList.append(captcha_name)
    random.seed(time.time())#设置随机数种子，当种子一样时生成相同的随机数
    random.shuffle(fileNameList)#shuffle将列表重新随机排序。
    for i in range(TEST_IMAGE_NUMBER):
        name=fileNameList[i]
        print(CAPTCHA_IMAGE_PATH+name,TEST_IMAGE_PATH+name)
        shutil.move(CAPTCHA_IMAGE_PATH+name,TEST_IMAGE_PATH+name)

if __name__ == '__main__':
    # generate_captcha_image(CHAR_SET,CHAR_SET_LEN,CAPTCHA_IMAGE_PATH)
    prepare_test_set()
    sys.stdout.write("\nFinished")
    sys.stdout.flush()




