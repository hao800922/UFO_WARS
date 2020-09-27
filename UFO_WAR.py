#!/usr/bin/env python
# coding: utf-8

# In[3]:


# tensorflow 2.3
# cv2.VideoCapture(0) 不一樣的話記得更換

from pygame.locals import MOUSEBUTTONDOWN ,KEYDOWN, K_ESCAPE, K_q, K_x, K_f, K_z, K_c, K_v, FULLSCREEN
import pygame
from pygame.locals import Color, QUIT, MOUSEBUTTONDOWN, USEREVENT, USEREVENT
from pygame.sprite import collide_rect, Sprite, spritecollide
import cv2
import sys
import tensorflow as tf
import numpy as np
import time
import random
import os
import sqlite3
from datetime import datetime,timezone,timedelta

# 連結Database
db = sqlite3.connect('game.db')
cursor = db.cursor()

# 讀取User資訊
with open('User.txt','r') as f:
    test = f.read().splitlines()
User = test[0]

# 視窗大小
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# ufo、戰機圖片大小
IMAGEWIDTH = 150
IMAGEHEIGHT = 100

# 雷射大小
LASERWIDTH = 10
LASERHEIGHT = WINDOW_HEIGHT

# 按鈕大小
BUTTONWIDTH = 256
BUTTONHEIGHT = 128

# 閃電大小
LIGHTWIDTH = 1280
LIGHTHEIGHT = 720


# 爆炸大小
BUMPWIDTH = 600
BUMPHEIGHT = 600

ufo_team = pygame.sprite.Group() 

# tf姿勢預測部分
####################################################################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def parse_output(heatmap_data,offset_data):
    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num,3),np.uint8)
    for i in range(heatmap_data.shape[-1]):
        joint_heatmap = heatmap_data[...,i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
        pose_kps[i,0] = int((remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i]))
        pose_kps[i,1] = int((remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num]))
        pose_kps[i,2] = (sigmoid(heatmap_data[:,:,i]).sum())*100/3
    return pose_kps



model = tf.lite.Interpreter('UFO_WAR/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

def post_detection(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1) # 左右對調
            
            

    input_img = tf.reshape(tf.image.resize(frame[:,:,::-1], [257,257]), [1,257,257,3])
    floating_model = input_details[0]['dtype'] == np.float32
    if floating_model:
        input_img = (np.float32(input_img) - 127.5) / 127.5
    model.set_tensor(input_details[0]['index'], input_img)
    start = time.time()
    model.invoke()

    output_data =  model.get_tensor(output_details[0]['index'])
    offset_data = model.get_tensor(output_details[1]['index'])
    heatmaps = np.squeeze(output_data)
    offsets = np.squeeze(offset_data)

    show_img = np.squeeze((input_img.copy()*127.5+127.5)/255.0)[:,:,::-1]
    show_img = np.array(show_img*255,np.uint8)
    kps = parse_output(heatmaps,offsets)
            
    # 將webcam兩軸互換 ( np.array[y,x]=>x,y )
    frame = frame.swapaxes(0, 1)
    return frame,kps
###################################################################################################

# ufo部分
##################################################################################################
class UFO(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = random.randrange(0, 1280-IMAGEWIDTH)
        self.y = random.randrange(-20-IMAGEHEIGHT, 0)
        self.raw_image = pygame.image.load('UFO_WAR/UFO.png')
        self.image = pygame.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)
        self.width = IMAGEWIDTH
        self.height = IMAGEHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        ufo_team.add(self)
        
    def draw(self, win):
        win.blit(self.image, self.rect)
    def move(self):
        self.x += random.randrange(-10, 10)
        self.y += 10
        self.rect.topleft = (self.x, self.y)
###############################################################################################

# 時區轉換
###############################################################################################
def TW_Time(T):
    dt1 = datetime.strptime(T, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    return dt2.strftime("%Y-%m-%d %H:%M:%S")
###############################################################################################

# 戰機部分
###############################################################################################
class Bonber(pygame.sprite.Sprite):
    def __init__(self,  x, y):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load('UFO_WAR/bonber.png')
        self.image = pygame.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = IMAGEWIDTH
        self.height = IMAGEHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
###############################################################################################
bonber = Bonber(0, 0)

# 雷射部分
################################################################################################
class LaserB(pygame.sprite.Sprite):
    def __init__(self,  x, y):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load('UFO_WAR/bluebeam.png')
        self.image = pygame.transform.scale(self.raw_image, (LASERWIDTH, LASERHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.midbottom = (self.x, self.y)
        self.width = LASERWIDTH
        self.height = LASERHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
################################################################################################


# 閃電部分
################################################################################################
class Light(pygame.sprite.Sprite):
    def __init__(self,  x, y):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load('UFO_WAR/lightning.png')
        self.image = pygame.transform.scale(self.raw_image, (LIGHTWIDTH, LIGHTHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = LIGHTWIDTH
        self.height = LIGHTHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
################################################################################################

# 爆炸部分
################################################################################################
class Bump(pygame.sprite.Sprite):
    def __init__(self,  x, y):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load('UFO_WAR/bump.png')
        self.image = pygame.transform.scale(self.raw_image, (BUMPWIDTH, BUMPHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = BUMPWIDTH
        self.height = BUMPHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
################################################################################################

# 按鈕部分
################################################################################################
class Button(pygame.sprite.Sprite):
    def __init__(self, x, y, picture):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load(picture)
        self.image = pygame.transform.scale(self.raw_image, (BUTTONWIDTH, BUTTONHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = BUTTONWIDTH
        self.height = BUTTONHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
################################################################################################
button_start = Button(640, 360, 'UFO_WAR/start.png')
button_exit = Button(1100, 650, 'UFO_WAR/exit.png')
button_demo = Button(290, 360, 'UFO_WAR/demo.png')
button_score = Button(990, 360, 'UFO_WAR/score.png')
button_menu = Button(180, 650, 'UFO_WAR/menu.png')
button_again = Button(640, 650, 'UFO_WAR/again.png')



# 鏡頭選擇
camera = cv2.VideoCapture(0)
camera_x, camera_y = (1280, 720)
camera.set(cv2.CAP_PROP_FRAME_WIDTH , camera_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_y)

# 起始背景
bg = pygame.image.load('UFO_WAR/bg.jpg')
bg_score = pygame.image.load('UFO_WAR/bg_score.png')





def main():
           
    try:
        
        # 初始化
        pygame.init()
        
        # 視窗畫面大小
        window_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # 遊戲名稱
        pygame.display.set_caption('UFO WAR')

        # 遊戲icon
        logo = pygame.image.load('UFO_WAR/alien.png')
        pygame.display.set_icon(logo)
        
        # 起始畫面白色
        window_surface.fill([255, 255, 255])
        
        # 導入全螢幕設定
        Fullscreen = False

        
        
        # ufo參數設定    
        ufo = UFO()
        ufo2 = UFO()
        ufo3 = UFO()
        ufo4 = UFO()
        ufo5 = UFO()
        
        # 戰機參數設定         
        bonber = Bonber(0, 0)
        
        
        
        
        
        
# 事件設定
#################################################################################
        # 共三件事 
        # 1. ufo移動 
        reload_ufo_event = USEREVENT+1
        pygame.time.set_timer(reload_ufo_event, 200) # 設定每300毫秒更新一次
        
        # 2. 撞到ufo
        kill_event = USEREVENT+2
        pygame.time.set_timer(kill_event, 60) # 設定每100毫秒更新一次
        
        # 3. 碰撞按鈕 (已關閉)
        select_button = USEREVENT+3
        pygame.time.set_timer(select_button, 100) # 設定每100毫秒更新一次
#################################################################################

        # 分數起始值
        points = 0
        
        # 時間起始值
        T = 0
        
        # 各文字字形與大小設定
        my_font = pygame.font.Font('chinese.msyh.ttf', 60)
        my_hit_font = pygame.font.Font('chinese.msyh.ttf', 240)
        
        
# 進入Pygame      
###############################################################################################################        
        On_status = True
        Menu = True
        while On_status:
            bonber = Bonber(0, 0)
            
# 進入選單 
###################################################################
            
            while Menu:

                # 加上背景圖片
                window_surface.blit(bg, (0, 0))
                
                # 加上按鈕
                button_start.draw(window_surface)
                button_exit.draw(window_surface)
                button_demo.draw(window_surface)
                button_score.draw(window_surface)

# 碰撞按鈕 (已關閉)                
#                 bonber = Bonber(0, 0)

#                 # 鏡頭讀取
#                 ret, frame = camera.read()

#                 # 姿勢預測
#                 frame,kps = post_detection(frame)
#                 # tf姿勢預測結果 (這邊只放鼻子)
#                 nose_Score = kps[0,2]
#                 nose_x = int(kps[0,1]*1280/257)
#                 nose_y = int(kps[0,0]*720/257)


#                 # tf姿勢預測結果超過閾值，再將其畫入畫面
#                 if nose_Score >= 30:
#                     bonber = Bonber(nose_x, nose_y)
#                     window_surface.blit(bonber.image, bonber.rect)


                # 滑鼠位置偵測
                mouse_pos = pygame.mouse.get_pos()



                #事件迴圈
                for event in pygame.event.get():

                    # 關閉事件 (按下右上叉叉，離開遊戲)
                    if event.type == pygame.QUIT:
                        On_status = False
                        sys.exit(0)
                        camera.release()

                    # 鍵盤事件
                    elif event.type == KEYDOWN:
                        
                        # 按下ESC or v，離開遊戲
                        if event.key == K_ESCAPE or event.key == K_v:
                            On_status = False
                            sys.exit(0)
                            camera.release()
                        
                        # 按下x，開始遊戲
                        elif event.key == K_x:
                            Menu = False
                            Run = True
                            Score = False
                            Demo = False
                            
                        # 按下z，開始Demo    
                        elif event.key == K_z:
                            Menu = False
                            Run = False
                            Score = False
                            Demo = True
                        
                        # 按下c，積分頁面
                        elif event.key == K_c:
                            Menu = False
                            Run = False
                            Score = True
                            Demo = False
                        
                        # 按下F，切換全螢幕
                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)
                    
                    # 滑鼠事件
                    elif event.type == MOUSEBUTTONDOWN:
                        
                        # 按下start按鈕，開始遊戲
                        if button_start.rect.collidepoint(mouse_pos):
                            Menu = False
                            Run = True
                            Score = False
                            Demo = False

                        # 按下score按鈕，積分頁面
                        elif button_score.rect.collidepoint(mouse_pos):
                            Menu = False
                            Run = False
                            Score = True
                            Demo = False  
                            
                        # 按下demo按鈕，開始Demo    
                        elif button_demo.rect.collidepoint(mouse_pos):
                            Menu = False
                            Run = False
                            Score = False
                            Demo = True                           
                        
                        # 按下exit按鈕，離開遊戲
                        elif button_exit.rect.collidepoint(mouse_pos):
                            On_status = False
                            sys.exit(0)
                            camera.release()

# 碰撞按鈕 (已關閉)
#                     elif event.type == select_button:
#                         if pygame.sprite.collide_rect(button_start, bonber):
#                             Menu=False
#                             Run = True
#                             Score = False
#                             Demo = False

#                         elif pygame.sprite.collide_rect(button_start, bonber):
#                             Menu=False
#                             Run = False
#                             Score = True
#                             Demo = False
                            
#                         elif pygame.sprite.collide_rect(button_start, bonber):
#                             Menu=False
#                             Run = False
#                             Score = False
#                             Demo = True
 
#                         elif pygame.sprite.collide_rect(button_exit, bonber):
#                             On_status = False
#                             sys.exit(0)
#                             camera.release()
                
                # 循環更新畫面
                pygame.display.update()

###################################################################            


# 開始遊戲
###################################################################
  
            while Run:
                

                # 鏡頭讀取
                ret, frame = camera.read()

                # 姿勢預測
                frame,kps = post_detection(frame)

                # 將webcam影像當作背景
                pygame.surfarray.blit_array(window_surface, frame)

                # tf姿勢預測結果 (這邊只放鼻子)
                nose_Score = kps[0,2]
                nose_x = int(kps[0,1]*1280/257)
                nose_y = int(kps[0,0]*720/257)


                # tf姿勢預測結果超過閾值，再將其畫入畫面
                if nose_Score >= 30:
                    bonber = Bonber(nose_x, nose_y)

                    # 雷射設定
                    LlaserB = LaserB(nose_x-IMAGEWIDTH//2, nose_y)
                    RlaserB = LaserB(nose_x+IMAGEWIDTH//2, nose_y)

                    # 閃電設定
                    light = Light(nose_x,nose_y)

                    # 爆炸設定
                    bump = Bump(ufo.x+IMAGEWIDTH/2,ufo.y+IMAGEHEIGHT/2)
                    bump2 = Bump(ufo2.x+IMAGEWIDTH/2,ufo2.y+IMAGEHEIGHT/2)
                    bump3 = Bump(ufo3.x+IMAGEWIDTH/2,ufo3.y+IMAGEHEIGHT/2)
                    bump4 = Bump(ufo4.x+IMAGEWIDTH/2,ufo4.y+IMAGEHEIGHT/2)
                    bump5 = Bump(ufo5.x+IMAGEWIDTH/2,ufo5.y+IMAGEHEIGHT/2)
                    bump0 = Bump(nose_x,nose_y)

                    # 放入戰機
                    window_surface.blit(bonber.image, bonber.rect)

                # 將UFO放入畫面
                ufo.draw(window_surface)
                ufo2.draw(window_surface)
                ufo3.draw(window_surface)
                ufo4.draw(window_surface)
                ufo5.draw(window_surface)

                # 正式處理事件
                for event in pygame.event.get():



                    # 關閉事件1 (按下右上叉叉，離開遊戲)
                    if event.type == pygame.QUIT:
                        On_status = False
                        sys.exit(0)
                        camera.release()

                    # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_v:
                            On_status = False
                            sys.exit(0)
                            camera.release()
                            
                        elif event.key == K_z:

                            Menu = True
                            Run = False
                            Score = False
                            Demo = False

                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)

                        elif event.key == K_x:
                            LlaserB.draw(window_surface)
                            RlaserB.draw(window_surface)
                            if pygame.sprite.collide_rect(ufo, LlaserB):
                                bump.draw(window_surface)
                                ufo.kill()
                                ufo = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo, RlaserB):
                                bump.draw(window_surface)
                                ufo.kill()
                                ufo = UFO()
                                points += 1 # 分數累計
                                
                            if pygame.sprite.collide_rect(ufo2, LlaserB):
                                bump2.draw(window_surface)
                                ufo2.kill()
                                ufo2 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo2, RlaserB):
                                bump2.draw(window_surface)
                                ufo2.kill()
                                ufo2 = UFO()
                                points += 1 # 分數累計

                            if pygame.sprite.collide_rect(ufo3, LlaserB):
                                bump3.draw(window_surface)
                                ufo3.kill()
                                ufo3 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo3, RlaserB):
                                bump3.draw(window_surface)
                                ufo3.kill()
                                ufo3 = UFO()
                                points += 1 # 分數累計

                            if pygame.sprite.collide_rect(ufo4, LlaserB):
                                bump4.draw(window_surface)
                                ufo4.kill()
                                ufo4 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo4, RlaserB):
                                bump4.draw(window_surface)
                                ufo4.kill()
                                ufo4 = UFO()
                                points += 1 # 分數累計

                            if pygame.sprite.collide_rect(ufo5, LlaserB):
                                bump5.draw(window_surface)
                                ufo5.kill()
                                ufo5 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo5, RlaserB):
                                bump5.draw(window_surface)
                                ufo5.kill()
                                ufo5 = UFO()
                                points += 1 # 分數累計
                                
                        elif event.key == K_q:
                            light.draw(window_surface)
                            
                            if pygame.sprite.collide_rect(ufo, light):
                                bump.draw(window_surface)
                                ufo.kill()
                                ufo = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo2, light):
                                bump2.draw(window_surface)
                                ufo2.kill()
                                ufo2 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo3, light):
                                bump3.draw(window_surface)
                                ufo3.kill()
                                ufo3 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo4, light):
                                bump4.draw(window_surface)
                                ufo4.kill()
                                ufo4 = UFO()
                                points += 1  # 分數累計

                            if pygame.sprite.collide_rect(ufo5, light):
                                bump5.draw(window_surface)
                                ufo5.kill()
                                ufo5 = UFO()
                                points += 1  # 分數累計



                    # ufo移動 (我這邊是直接殺死一隻，在創造一隻，製造位移假象)
                    elif event.type == reload_ufo_event:
                        ufo.move()
                        ufo2.move()
                        ufo3.move()
                        ufo4.move()
                        ufo5.move()

                    


                    # 殺死ufo
                    elif event.type == kill_event:
                        if pygame.sprite.spritecollide(bonber, ufo_team, False):
                            bump0.draw(window_surface)
                            print(points)
                            print(User)
                            cursor.execute('INSERT INTO `ufo_war`( `name`, `score`) VALUES (?,?)',[User,points])
                            db.commit()
                            ufo.kill()
                            ufo = UFO()
                            ufo2.kill()
                            ufo2 = UFO()
                            ufo3.kill()
                            ufo3 = UFO()
                            ufo4.kill()
                            ufo4 = UFO()
                            ufo5.kill()
                            ufo5 = UFO()

                            points = 0
                            T = 0
                            Menu = False
                            Run = False
                            Score = True
                            Demo = False
                                                

                T += 0.2 # 時間累計 


                # 遊戲分數與時間 內文 與 色彩 
                text_surface = my_font.render(f'Points: {format(points)}', True, (0, 255, 0))
                text_time = my_font.render(f'time: {format(round(T))}', True, (0, 255, 0))


                # 正式放入 文字
                window_surface.blit(text_surface, (10, 0))
                window_surface.blit(text_time, (10, 40))

                # 循環更新
                pygame.display.update()

#############################################################

# 進入評分
#############################################################
            
            while Score:

                window_surface.fill([255, 255, 255])
                window_surface.blit(bg_score, (0, 0))


                button_menu.draw(window_surface)
                button_again.draw(window_surface)
                button_exit.draw(window_surface)

                CS = list(cursor.execute('SELECT * FROM `ufo_war` WHERE `name` = ? Order BY `time` DESC',[User]))
                if len(CS):
                    current_score = my_font.render(f'目前分數:   {CS[0][2]}分   {TW_Time(CS[0][3])[:10]}', True, (0, 255, 0))
                    window_surface.blit(current_score, (200, 70))

                HS = list(cursor.execute('SELECT * FROM `ufo_war` WHERE `name` = ? Order BY `score` DESC',[User]))
                if len(CS):
                    highest_score = my_font.render(f'最高分數:   {HS[0][2]}分   {TW_Time(HS[0][3])[:10]}', True, (255, 127, 80))
                    window_surface.blit(highest_score, (200, 150))

                if len(CS) >= 4:
                    History = my_font.render('歷史分數', True, (30, 144, 255))
                    window_surface.blit(History, (200, 230))
                    for i in range(1,4):
                        History_score = my_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
                        window_surface.blit(History_score, (200, 230+80*i))
                elif len(CS) == 3:
                    History = my_font.render('歷史分數', True, (30, 144, 255))
                    window_surface.blit(History, (200, 230))
                    for i in range(1,3):
                        History_score = my_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
                        window_surface.blit(History_score, (200, 230+80*i))
                elif len(CS) == 2:
                    History = my_font.render('歷史分數', True, (30, 144, 255))
                    window_surface.blit(History, (200, 230))
                    for i in range(1,2):
                        History_score = my_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
                        window_surface.blit(History_score, (200, 230+80*i))

#                 # 鏡頭讀取
#                 ret, frame = camera.read()

#                 # 姿勢預測
#                 frame,kps = post_detection(frame)


#                 # tf姿勢預測結果 (這邊只放鼻子)
#                 nose_Score = kps[0,2]
#                 nose_x = int(kps[0,1]*1280/257)
#                 nose_y = int(kps[0,0]*720/257)


#                 # tf姿勢預測結果超過閾值，再將其畫入畫面
#                 if nose_Score >= 30:
#                     bonber = Bonber(nose_x, nose_y)
#                     window_surface.blit(bonber.image, bonber.rect)



                mouse_pos = pygame.mouse.get_pos()




                for event in pygame.event.get():

                    # 關閉事件1 (按下右上叉叉，離開遊戲)
                    if event.type == pygame.QUIT:
                        On_status = False
                        sys.exit(0)
                        camera.release()

                    # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_v:
                            On_status = False
                            sys.exit(0)
                            camera.release()

                        elif event.key == K_z:
                            Menu = True
                            Run = False
                            Score = False
                            Demo = False
                            
                        elif event.key == K_x:
                            Menu = False
                            Run = True
                            Score = False
                            Demo = False

                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)

                    elif event.type == MOUSEBUTTONDOWN:
                        if button_menu.rect.collidepoint(mouse_pos):
                            Menu = True
                            Run = False
                            Score = False
                            Demo = False

                        elif button_again.rect.collidepoint(mouse_pos):
                            Menu = False
                            Run = True
                            Score = False
                            Demo = False
                            
                        elif button_exit.rect.collidepoint(mouse_pos):
                            On_status = False
                            sys.exit(0)
                            camera.release()
                            
#                     elif event.type == select_button:
#                         if pygame.sprite.collide_rect(button_menu, bonber):
#                             Menu = True
#                             Run = False
#                             Score = False
#                             Demo = False
                            
#                         elif pygame.sprite.collide_rect(button_again, bonber):
#                             Menu = False
#                             Run = True
#                             Score = False
#                             Demo = False    
                            
#                         elif pygame.sprite.collide_rect(button_exit, bonber):
#                             On_status = False
#                             sys.exit(0)
#                             camera.release()

                pygame.display.update()

#############################################################

# 進入demo
#############################################################
            demo_video = cv2.VideoCapture('UFO_WAR/DEMO_UFO_WARS.mp4')
            while Demo:
                
                hasFrame, img = demo_video.read()
                time.sleep(0.001)    
                if not hasFrame:
                    Menu = True
                    Run = False
                    Score = False
                    Demo = False
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.swapaxes(0, 1)
        
        
                pygame.surfarray.blit_array(window_surface, img)
        
                button_menu.draw(window_surface)
                
                



                

#                 # 鏡頭讀取
#                 ret, frame = camera.read()

#                 # 姿勢預測
#                 frame,kps = post_detection(frame)

#                 # tf姿勢預測結果 (這邊只放鼻子)
#                 nose_Score = kps[0,2]
#                 nose_x = int(kps[0,1]*1280/257)
#                 nose_y = int(kps[0,0]*720/257)


#                 # tf姿勢預測結果超過閾值，再將其畫入畫面
#                 if nose_Score >= 30:
#                     bonber = Bonber(nose_x, nose_y)
#                     window_surface.blit(bonber.image, bonber.rect)



                mouse_pos = pygame.mouse.get_pos()
                

                for event in pygame.event.get():

                    # 關閉事件1 (按下右上叉叉，離開遊戲)
                    if event.type == pygame.QUIT:
                        On_status = False
                        sys.exit(0)
                        camera.release()

                    # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            On_status = False
                            sys.exit(0)
                            camera.release()

                        elif event.key == K_x:
                            Menu = True
                            Run = False
                            Score = False
                            Demo = False

                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)

                    elif event.type == MOUSEBUTTONDOWN:
                        if button_menu.rect.collidepoint(mouse_pos):
                            Menu = True
                            Run = False
                            Score = False
                            Demo = False
                            


                            
#                     elif event.type == select_button:
#                         if pygame.sprite.collide_rect(button_menu, bonber):
#                             Menu = True
#                             Run = False
#                             Score = False
#                             Demo = False
                            

                pygame.display.update()

#############################################################

    except (KeyboardInterrupt, SystemExit):
        pygame.quit()
        cv2.destroyAllWindows()
        camera.release()
        
if __name__ == '__main__':    
    main()

