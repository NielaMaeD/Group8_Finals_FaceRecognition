<div style="text-align: center;">
  <div align="center">
    <h1>𝗠𝗮𝗿𝘃𝗲𝗹 𝗖𝗶𝗻𝗲𝗺𝗮𝘁𝗶𝗰 𝗨𝗻𝗶𝘃𝗲𝗿𝘀𝗲 𝗕𝗼𝘅 𝗢𝗳𝗳𝗶𝗰𝗲 𝗙𝗮𝗰𝗲 𝗥𝗲𝗰𝗼𝗴𝗻𝗶𝘁𝗶𝗼𝗻</h1>
  </div>
  </div>
  
<div align="center">
  <div style="font-style: Autor one;">
    <sup>𝖠𝗎𝗍𝗁𝗈𝗋𝗌: ɴɪᴇʟᴀ ᴍᴀᴇ ᴅɪᴍᴀᴄᴜʟᴀɴɢᴀɴ, ᴄʟᴀʀᴇɴᴄᴇ ᴊᴏʏ ɪʟᴀɢᴀɴ, ᴀɴᴅ ᴀᴘʀɪʟ ᴍᴇʀᴄᴀᴅᴏ</sup>
    <br>
    <sup>𝖳𝗁𝗂𝗌 𝗋𝖾𝗉𝗈𝗌𝗂𝗍𝗈𝗋𝗒 𝖼𝗈𝗇𝗍𝖺𝗂𝗇𝗌 𝖿𝖺𝖼𝗂𝖺𝗅 𝗋𝖾𝖼𝗈𝗀𝗇𝗂𝗍𝗂𝗈𝗇 𝖽𝖺𝗍𝖺 for 𝖼𝗁𝖺𝗋𝖺𝖼𝗍𝖾𝗋𝗌 𝖿𝗋𝗈𝗆 𝗍𝗁𝖾 𝖬𝖺𝗋𝗏𝖾𝗅 𝖢𝗂𝗇𝖾𝗆𝖺𝗍𝗂𝖼 𝖴𝗇𝗂𝗏𝖾𝗋𝗌𝖾 (MCU), specifically the Avengers.</sup>
  </div>
</div>  

![368129771_376256858089592_233386268408499940_n](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/70c175c3-2a97-465a-a013-3b3efd10a7b7)
_When it comes to blockbuster cinema, the Marvel Cinematic Universe (MCU) stands as a groundbreaking achievement, reshaping the landscape of superhero storytelling with its interconnected narratives, iconic characters, and an overarching vision that extends far beyond the boundaries of conventional filmmaking. 
The MCU officially began with the release of "Iron Man" in 2008 and since then, Marvel Studios has produced a series of interconnected films, grouped into phases, each contributing to an overarching storyline. These phases often culminate in ensemble films, such as **"The Avengers,"** where multiple superheroes come together to face a common threat. The success of the MCU has led to a cultural phenomenon, with a dedicated fanbase and widespread acclaim for its ability to bring beloved comic book characters to life on the big screen._

---

<div style="text-align: center;">
  <div align="center">
  <h1>ꜰᴀᴄᴇ ʀᴇᴄᴏɢɴɪᴛɪᴏɴ ᴄᴏᴅᴇꜱ</h1>
  </div>
</div>  



_<div align = "center">
  <sup>For this activity, the group is tasked to run a face recognition program that will identify unknown and known faces. This will feature the identification of certain characters from the Avengers, as well as faces that are unknown based on the limited set of known images.</sup>_
</div>
_<sup>The codes that will be used in accomplishing this task are listed below:</sup>_

---

**</>𝗖𝗹𝗼𝗻𝗲 𝗥𝗲𝗽𝗼𝘀𝗶𝘁𝗼𝗿𝗶𝗲𝘀 𝗮𝗻𝗱 𝗗𝗼𝘄𝗻𝗹𝗼𝗮𝗱 𝗗𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝗰𝗶𝗲𝘀**

    !git clone https://github.com/NielaMaeD/Group8_Finals_FaceRecognition.git
    !pip install face_recognition
    %cd Group8_Finals_FaceRecognition
---

**</>𝗘𝗻𝗰𝗼𝗱𝗶𝗻𝗴 𝗣𝗿𝗼𝗳𝗶𝗹𝗲𝘀 𝗨𝘀𝗶𝗻𝗴 𝗞𝗻𝗼𝘄𝗻 𝗙𝗮𝗰𝗲 𝗜𝗺𝗮𝗴𝗲𝘀**

    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2

    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("Thor.jpg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]
    
    face_2 = face_recognition.load_image_file("Iron Man.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]
    
    face_3 = face_recognition.load_image_file("Black Widow.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]
    
    face_4 = face_recognition.load_image_file("Captain America.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]
    
    face_5 = face_recognition.load_image_file("Hawkeye.jpg")
    face_5_encoding = face_recognition.face_encodings(face_5)[0]
    
    face_6 = face_recognition.load_image_file("Hulk.jpg")
    face_6_encoding = face_recognition.face_encodings(face_6)[0]
    
    known_face_encodings = [
                            face_1_encoding,
                            face_2_encoding,
                            face_3_encoding,
                            face_4_encoding,
                            face_5_encoding,
                            face_6_encoding
    ]
    
    known_face_names = [
                        "Thor Odinson",
                        "Tony Stark",
                        "Natasha Romanoff",
                        "Steve Rogers",
                        "Clint Barton",
                        "Bruce Banner"
    ]
---
**</>𝗥𝘂𝗻𝗻𝗶𝗻𝗴 𝗼𝗳 𝗙𝗮𝗰𝗲 𝗥𝗲𝗰𝗼𝗴𝗻𝗶𝘁𝗶𝗼𝗻 𝗼𝗻 𝘁𝗵𝗲 𝗢𝗿𝗶𝗴𝗶𝗻𝗮𝗹 𝗖𝗮𝘀𝘁 𝗼𝗳 𝗔𝘃𝗲𝗻𝗴𝗲𝗿𝘀**

_🔳This Python code leverages the face_recognition library and OpenCV to identify the Avengers' original six members - Iron Man, Captain America, Thor, Hulk, Black Widow, and Hawkeye - in an image. It analyzes an unknown image, detects faces, and compares their encodings to a set of known encodings for these superheroes._

_🔳The code then draws rectangles around recognized faces, annotates them with the corresponding superhero names, and displays the modified image, showcasing the results of the face recognition process. This technology could be valuable for various applications, including identifying individuals entering secured areas or even monitoring superhero activity!_

    file_name = "Avengers Original 6.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Unknown"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),2)
      cv2.putText(unknown_image_to_draw,name, (left, bottom-20), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)

---

<div style="text-align: center;">
  <div align="center">
  <h1>ᴀᴠᴇɴɢᴇʀꜱ ᴏʀɪɢɪɴᴀʟ ꜱɪx</h1>
  </div>
</div>

![Avengers Original 6](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/1628d460-b7b8-4049-a84b-d12cf7e5617a)

_At the heart of this cinematic revolution lies the Avengers Original Six – Captain America, Iron Man, Thor, Hulk, Black Widow, and Hawkeye. Debuting in 2012 Marvel Studios' groundbreaking film "The Avengers", these founding heroes stand as a testament to the power of collaboration, individual growth, and the enduring spirit of heroism._

---

<div style="text-align: center;">
  <div align="center">
  <h1>ᴀᴠᴇɴɢᴇʀꜱ ᴏʀɪɢɪɴᴀʟ ꜱɪx ꜰᴀᴄᴇ ʀᴇᴄᴏɢɴɪᴛɪᴏɴ</h1>
  </div>
</div>

![download (1)](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/fbe00e0e-c503-41cf-a988-1cb7813c10a6)

<div style="text-align: center;">
  <div align="center">
  <sup>Facial recognition is a method of verifying or establishing an individual's identity by analyzing their facial features. This technology is employed to recognize individuals in photographs, videos, or real-time scenarios. It falls under the umbrella of biometric security, which encompasses other types of biometric software such as voice recognition, fingerprint recognition, and eye retina or iris recognition. </sup>
    <br>
  <sup>The process of face recognition generally involves capturing or acquiring an image of a person's face, extracting key facial features, and then comparing this information with a database of known faces. The goal is to determine whether the face in question matches any pre-existing records. While facial recognition is primarily utilized in security and law enforcement, there is a growing interest in exploring its applications in various other fields.</sup>
  </div>
</div>

---

<div align="center">
  
### **AVENGERS ORIGINAL SIX AND THEIR PORTRAYED HEROES**
### <sup>ᴋɴᴏᴡɴ x ꜰᴀᴄᴇ ʀᴇᴄᴏɢɴɪᴛɪᴏɴ</sup>

---
</div>

<details>
  <summary>
    <sub>FACE RECOGNITION CODE FOR KNOWN AND UNKNOWN</sub>
  </summary>

        file_name = " "
        unknown_image = face_recognition.load_image_file(file_name)
        unknown_image_to_draw = cv2.imread(file_name)
        
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        
        for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
          name = "Unknown"
        
          face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
          best_match_index = np.argmin(face_distances)
          if matches[best_match_index]:
            name = known_face_names[best_match_index]
          cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),2)
          cv2.putText(unknown_image_to_draw,name, (left, bottom-20), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2, cv2.LINE_AA)
        
        cv2_imshow(unknown_image_to_draw)
        
</details>

# **TONY STARK AS IRON MAN** ![ironman](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/ba549aed-fbaf-431a-873d-7dd2522d8ff5)

  <div align="justify">
𝘛𝘰𝘯𝘺 𝘚𝘵𝘢𝘳𝘬'𝘴 𝘥𝘶𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘢𝘴 𝘐𝘳𝘰𝘯 𝘔𝘢𝘯 𝘦𝘯𝘤𝘢𝘱𝘴𝘶𝘭𝘢𝘵𝘦𝘴 𝘢 𝘤𝘢𝘱𝘵𝘪𝘷𝘢𝘵𝘪𝘯𝘨 𝘦𝘹𝘱𝘭𝘰𝘳𝘢𝘵𝘪𝘰𝘯 𝘰𝘧 𝘳𝘦𝘢𝘭 𝘢𝘯𝘥 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘱𝘦𝘳𝘴𝘰𝘯𝘢𝘴. 𝘐𝘯 𝘩𝘪𝘴 𝘳𝘦𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘩𝘦 𝘪𝘴 𝘢 𝘣𝘳𝘪𝘭𝘭𝘪𝘢𝘯𝘵 𝘣𝘪𝘭𝘭𝘪𝘰𝘯𝘢𝘪𝘳𝘦, 𝘦𝘱𝘪𝘵𝘰𝘮𝘪𝘻𝘪𝘯𝘨 𝘸𝘦𝘢𝘭𝘵𝘩 𝘢𝘯𝘥 𝘪𝘯𝘵𝘦𝘭𝘭𝘦𝘤𝘵. 𝘏𝘰𝘸𝘦𝘷𝘦𝘳, 𝘢𝘴 𝘐𝘳𝘰𝘯 𝘔𝘢𝘯, 𝘚𝘵𝘢𝘳𝘬'𝘴 𝘢𝘳𝘮𝘰𝘳𝘦𝘥 𝘢𝘭𝘵𝘦𝘳 𝘦𝘨𝘰 𝘣𝘦𝘤𝘰𝘮𝘦𝘴 𝘢 𝘴𝘺𝘮𝘣𝘰𝘭𝘪𝘤 𝘦𝘹𝘵𝘦𝘯𝘴𝘪𝘰𝘯 𝘰𝘧 𝘩𝘪𝘴 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘴𝘩𝘰𝘸𝘤𝘢𝘴𝘪𝘯𝘨 𝘣𝘰𝘵𝘩 𝘩𝘪𝘴 𝘪𝘯𝘨𝘦𝘯𝘶𝘪𝘵𝘺 𝘢𝘯𝘥 𝘩𝘪𝘥𝘥𝘦𝘯 𝘷𝘶𝘭𝘯𝘦𝘳𝘢𝘣𝘪𝘭𝘪𝘵𝘪𝘦𝘴. 𝘛𝘩𝘦 𝘪𝘯𝘵𝘦𝘳𝘱𝘭𝘢𝘺 𝘣𝘦𝘵𝘸𝘦𝘦𝘯 𝘩𝘪𝘴 𝘱𝘶𝘣𝘭𝘪𝘤 𝘪𝘮𝘢𝘨𝘦 𝘢𝘯𝘥 𝘱𝘳𝘪𝘷𝘢𝘵𝘦 𝘴𝘵𝘳𝘶𝘨𝘨𝘭𝘦𝘴 𝘦𝘹𝘦𝘮𝘱𝘭𝘪𝘧𝘪𝘦𝘴 𝘵𝘩𝘦 𝘤𝘰𝘮𝘱𝘭𝘦𝘹𝘪𝘵𝘺 𝘰𝘧 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘪𝘯 𝘵𝘩𝘦 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘯𝘢𝘳𝘳𝘢𝘵𝘪𝘷𝘦, 𝘸𝘩𝘦𝘳𝘦 𝘵𝘩𝘦 𝘦𝘹𝘵𝘦𝘳𝘯𝘢𝘭 𝘧𝘢𝘤𝘢𝘥𝘦 𝘢𝘯𝘥 𝘱𝘦𝘳𝘴𝘰𝘯𝘢𝘭 𝘤𝘩𝘢𝘭𝘭𝘦𝘯𝘨𝘦𝘴 𝘤𝘰𝘢𝘭𝘦𝘴𝘤𝘦 𝘪𝘯𝘵𝘰 𝘢 𝘥𝘺𝘯𝘢𝘮𝘪𝘤 𝘢𝘯𝘥 𝘤𝘰𝘮𝘱𝘦𝘭𝘭𝘪𝘯𝘨 𝘤𝘩𝘢𝘳𝘢𝘤𝘵𝘦𝘳.   
  </div>

  <br>
  
_<sup>🔻Using the provided face recognition code and analyzing the image "IRONMAN.jpg," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/9a8a19bf-8116-4a72-9c0c-6feacbef6ba8">
</p>

---

# **THOR ODINSON AS THE GOD OF THUNDER** ![thor](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/0212eaf3-c5e2-4bc4-bda1-f40513bc9dd9)
 
  <div align="justify">
𝘛𝘩𝘰𝘳, 𝘵𝘩𝘦 𝘎𝘰𝘥 𝘰𝘧 𝘛𝘩𝘶𝘯𝘥𝘦𝘳, 𝘱𝘳𝘦𝘴𝘦𝘯𝘵𝘴 𝘢 𝘥𝘪𝘴𝘵𝘪𝘯𝘤𝘵𝘪𝘷𝘦 𝘦𝘹𝘱𝘭𝘰𝘳𝘢𝘵𝘪𝘰𝘯 𝘰𝘧 𝘳𝘦𝘢𝘭 𝘢𝘯𝘥 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺. 𝘐𝘯 𝘩𝘪𝘴 𝘳𝘦𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘛𝘩𝘰𝘳 𝘪𝘴 𝘢 𝘳𝘰𝘺𝘢𝘭 𝘈𝘴𝘨𝘢𝘳𝘥𝘪𝘢𝘯 𝘱𝘳𝘪𝘯𝘤𝘦, 𝘸𝘪𝘦𝘭𝘥𝘪𝘯𝘨 𝘣𝘰𝘵𝘩 𝘴𝘵𝘳𝘦𝘯𝘨𝘵𝘩 𝘢𝘯𝘥 𝘳𝘦𝘨𝘢𝘭 𝘢𝘶𝘵𝘩𝘰𝘳𝘪𝘵𝘺. 𝘏𝘰𝘸𝘦𝘷𝘦𝘳, 𝘩𝘪𝘴 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘵𝘳𝘢𝘯𝘴𝘤𝘦𝘯𝘥𝘴 𝘮𝘰𝘳𝘵𝘢𝘭 𝘤𝘰𝘯𝘧𝘪𝘯𝘦𝘴, 𝘢𝘴 𝘵𝘩𝘦 𝘮𝘪𝘨𝘩𝘵𝘺 𝘈𝘷𝘦𝘯𝘨𝘦𝘳 𝘢𝘯𝘥 𝘨𝘶𝘢𝘳𝘥𝘪𝘢𝘯 𝘰𝘧 𝘈𝘴𝘨𝘢𝘳𝘥. 𝘛𝘩𝘦 𝘦𝘯𝘤𝘩𝘢𝘯𝘵𝘦𝘥 𝘔𝘫𝘰𝘭𝘯𝘪𝘳, 𝘩𝘪𝘴 𝘪𝘤𝘰𝘯𝘪𝘤 𝘩𝘢𝘮𝘮𝘦𝘳, 𝘴𝘦𝘳𝘷𝘦𝘴 𝘢𝘴 𝘣𝘰𝘵𝘩 𝘢 𝘸𝘦𝘢𝘱𝘰𝘯 𝘢𝘯𝘥 𝘢 𝘴𝘺𝘮𝘣𝘰𝘭 𝘰𝘧 𝘩𝘪𝘴 𝘥𝘪𝘷𝘪𝘯𝘦 𝘱𝘳𝘰𝘸𝘦𝘴𝘴. 𝘛𝘩𝘰𝘳'𝘴 𝘤𝘩𝘢𝘳𝘢𝘤𝘵𝘦𝘳 𝘥𝘦𝘭𝘷𝘦𝘴 𝘪𝘯𝘵𝘰 𝘵𝘩𝘦 𝘪𝘯𝘵𝘦𝘳𝘴𝘦𝘤𝘵𝘪𝘰𝘯 𝘰𝘧 𝘨𝘰𝘥𝘭𝘺 𝘳𝘦𝘴𝘱𝘰𝘯𝘴𝘪𝘣𝘪𝘭𝘪𝘵𝘪𝘦𝘴 𝘢𝘯𝘥 𝘵𝘩𝘦 𝘳𝘦𝘭𝘢𝘵𝘢𝘣𝘭𝘦 𝘴𝘵𝘳𝘶𝘨𝘨𝘭𝘦𝘴 𝘰𝘧 𝘢𝘥𝘢𝘱𝘵𝘪𝘯𝘨 𝘵𝘰 𝘩𝘶𝘮𝘢𝘯 𝘦𝘹𝘱𝘦𝘳𝘪𝘦𝘯𝘤𝘦𝘴 𝘰𝘯 𝘌𝘢𝘳𝘵𝘩. 𝘛𝘩𝘦 𝘫𝘶𝘹𝘵𝘢𝘱𝘰𝘴𝘪𝘵𝘪𝘰𝘯 𝘰𝘧 𝘩𝘪𝘴 𝘈𝘴𝘨𝘢𝘳𝘥𝘪𝘢𝘯 𝘩𝘦𝘳𝘪𝘵𝘢𝘨𝘦 𝘢𝘯𝘥 𝘦𝘢𝘳𝘵𝘩𝘭𝘺 𝘦𝘯𝘤𝘰𝘶𝘯𝘵𝘦𝘳𝘴 𝘶𝘯𝘥𝘦𝘳𝘴𝘤𝘰𝘳𝘦𝘴 𝘵𝘩𝘦 𝘥𝘺𝘯𝘢𝘮𝘪𝘤 𝘯𝘢𝘵𝘶𝘳𝘦 𝘰𝘧 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘸𝘩𝘦𝘳𝘦 𝘵𝘩𝘦 𝘦𝘹𝘵𝘳𝘢𝘰𝘳𝘥𝘪𝘯𝘢𝘳𝘺 𝘢𝘯𝘥 𝘵𝘩𝘦 𝘩𝘶𝘮𝘢𝘯 𝘤𝘰𝘦𝘹𝘪𝘴𝘵 𝘪𝘯 𝘢 𝘩𝘢𝘳𝘮𝘰𝘯𝘪𝘰𝘶𝘴 𝘺𝘦𝘵 𝘤𝘰𝘮𝘱𝘭𝘦𝘹 𝘯𝘢𝘳𝘳𝘢𝘵𝘪𝘷𝘦. 
  </div>
  
<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "THOR.png," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/061572ab-95d6-4bec-a2a7-64fd6bc61324">
</p>

---

# **STEVE ROGERS AS CAPTAIN AMERICA**![captain_america (1)](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/1f66fefd-9eed-437b-a895-090e9e853c48)

  <div align="justify">
𝘊𝘢𝘱𝘵𝘢𝘪𝘯 𝘈𝘮𝘦𝘳𝘪𝘤𝘢, 𝘢𝘭𝘴𝘰 𝘬𝘯𝘰𝘸𝘯 𝘢𝘴 𝘚𝘵𝘦𝘷𝘦 𝘙𝘰𝘨𝘦𝘳𝘴, 𝘦𝘮𝘣𝘰𝘥𝘪𝘦𝘴 𝘢 𝘱𝘢𝘵𝘳𝘪𝘰𝘵𝘪𝘤 𝘢𝘯𝘥 𝘶𝘯𝘸𝘢𝘷𝘦𝘳𝘪𝘯𝘨 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺. 𝘐𝘯 𝘩𝘪𝘴 𝘳𝘦𝘢𝘭 𝘱𝘦𝘳𝘴𝘰𝘯𝘢, 𝘙𝘰𝘨𝘦𝘳𝘴 𝘪𝘴 𝘢 𝘴𝘺𝘮𝘣𝘰𝘭 𝘰𝘧 𝘳𝘦𝘴𝘪𝘭𝘪𝘦𝘯𝘤𝘦, 𝘩𝘢𝘷𝘪𝘯𝘨 𝘵𝘳𝘢𝘯𝘴𝘧𝘰𝘳𝘮𝘦𝘥 𝘧𝘳𝘰𝘮 𝘢 𝘧𝘳𝘢𝘪𝘭 𝘺𝘰𝘶𝘯𝘨 𝘮𝘢𝘯 𝘪𝘯𝘵𝘰 𝘢 𝘴𝘶𝘱𝘦𝘳-𝘴𝘰𝘭𝘥𝘪𝘦𝘳 𝘵𝘩𝘳𝘰𝘶𝘨𝘩 𝘵𝘩𝘦 𝘚𝘶𝘱𝘦𝘳 𝘚𝘰𝘭𝘥𝘪𝘦𝘳 𝘚𝘦𝘳𝘶𝘮. 𝘈𝘴 𝘊𝘢𝘱𝘵𝘢𝘪𝘯 𝘈𝘮𝘦𝘳𝘪𝘤𝘢, 𝘩𝘦 𝘣𝘦𝘤𝘰𝘮𝘦𝘴 𝘵𝘩𝘦 𝘭𝘪𝘷𝘪𝘯𝘨 𝘦𝘮𝘣𝘰𝘥𝘪𝘮𝘦𝘯𝘵 𝘰𝘧 𝘫𝘶𝘴𝘵𝘪𝘤𝘦, 𝘸𝘪𝘦𝘭𝘥𝘪𝘯𝘨 𝘩𝘪𝘴 𝘪𝘤𝘰𝘯𝘪𝘤 𝘴𝘩𝘪𝘦𝘭𝘥 𝘢𝘴 𝘢 𝘴𝘺𝘮𝘣𝘰𝘭 𝘰𝘧 𝘩𝘰𝘱𝘦. 𝘙𝘰𝘨𝘦𝘳𝘴' 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘪𝘴 𝘥𝘦𝘦𝘱𝘭𝘺 𝘳𝘰𝘰𝘵𝘦𝘥 𝘪𝘯 𝘱𝘳𝘪𝘯𝘤𝘪𝘱𝘭𝘦𝘴 𝘰𝘧 𝘩𝘰𝘯𝘰𝘳, 𝘥𝘶𝘵𝘺, 𝘢𝘯𝘥 𝘴𝘢𝘤𝘳𝘪𝘧𝘪𝘤𝘦, 𝘳𝘦𝘧𝘭𝘦𝘤𝘵𝘪𝘯𝘨 𝘵𝘩𝘦 𝘪𝘥𝘦𝘢𝘭𝘴 𝘰𝘧 𝘢 𝘣𝘺𝘨𝘰𝘯𝘦 𝘦𝘳𝘢. 𝘛𝘩𝘦 𝘴𝘵𝘢𝘳-𝘴𝘱𝘢𝘯𝘨𝘭𝘦𝘥 𝘩𝘦𝘳𝘰 𝘴𝘵𝘢𝘯𝘥𝘴 𝘢𝘴 𝘢 𝘴𝘦𝘯𝘵𝘪𝘯𝘦𝘭 𝘢𝘨𝘢𝘪𝘯𝘴𝘵 𝘵𝘺𝘳𝘢𝘯𝘯𝘺, 𝘴𝘩𝘰𝘸𝘤𝘢𝘴𝘪𝘯𝘨 𝘵𝘩𝘦 𝘦𝘯𝘥𝘶𝘳𝘪𝘯𝘨 𝘴𝘵𝘳𝘦𝘯𝘨𝘵𝘩 𝘰𝘧 𝘤𝘩𝘢𝘳𝘢𝘤𝘵𝘦𝘳 𝘢𝘯𝘥 𝘶𝘯𝘸𝘢𝘷𝘦𝘳𝘪𝘯𝘨 𝘤𝘰𝘮𝘮𝘪𝘵𝘮𝘦𝘯𝘵 𝘵𝘰 𝘵𝘩𝘦 𝘨𝘳𝘦𝘢𝘵𝘦𝘳 𝘨𝘰𝘰𝘥. 𝘊𝘢𝘱𝘵𝘢𝘪𝘯 𝘈𝘮𝘦𝘳𝘪𝘤𝘢'𝘴 𝘥𝘶𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘪𝘦𝘴 𝘦𝘹𝘦𝘮𝘱𝘭𝘪𝘧𝘺 𝘵𝘩𝘦 𝘵𝘳𝘢𝘯𝘴𝘧𝘰𝘳𝘮𝘢𝘵𝘪𝘷𝘦 𝘱𝘰𝘸𝘦𝘳 𝘰𝘧 𝘤𝘰𝘶𝘳𝘢𝘨𝘦 𝘢𝘯𝘥 𝘵𝘩𝘦 𝘦𝘯𝘥𝘶𝘳𝘪𝘯𝘨 𝘭𝘦𝘨𝘢𝘤𝘺 𝘰𝘧 𝘢 𝘩𝘦𝘳𝘰 𝘸𝘩𝘰 𝘵𝘳𝘢𝘯𝘴𝘤𝘦𝘯𝘥𝘴 𝘵𝘪𝘮𝘦.
  </div>
  
<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "CAPTAIN AMERICA.png," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/1dffa6d9-7e92-4c35-8638-0d27c1c66c1a">
</p>

---

# **BRUCE BANNER AS THE HULK**![hulk_rage](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/ef7c66f0-d8ce-4e1e-a641-0cfec15bbece)


  <div align="justify">
𝘛𝘩𝘦 𝘏𝘶𝘭𝘬, 𝘢𝘭𝘵𝘦𝘳 𝘦𝘨𝘰 𝘰𝘧 𝘋𝘳. 𝘉𝘳𝘶𝘤𝘦 𝘉𝘢𝘯𝘯𝘦𝘳, 𝘳𝘦𝘱𝘳𝘦𝘴𝘦𝘯𝘵𝘴 𝘢 𝘶𝘯𝘪𝘲𝘶𝘦 𝘥𝘶𝘢𝘭𝘪𝘵𝘺 𝘪𝘯 𝘵𝘩𝘦 𝘳𝘦𝘢𝘭𝘮 𝘰𝘧 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘪𝘦𝘴. 𝘐𝘯 𝘩𝘪𝘴 𝘳𝘦𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘉𝘢𝘯𝘯𝘦𝘳 𝘪𝘴 𝘢 𝘣𝘳𝘪𝘭𝘭𝘪𝘢𝘯𝘵 𝘴𝘤𝘪𝘦𝘯𝘵𝘪𝘴𝘵 𝘨𝘳𝘢𝘱𝘱𝘭𝘪𝘯𝘨 𝘸𝘪𝘵𝘩 𝘵𝘩𝘦 𝘤𝘰𝘯𝘴𝘦𝘲𝘶𝘦𝘯𝘤𝘦𝘴 𝘰𝘧 𝘨𝘢𝘮𝘮𝘢 𝘳𝘢𝘥𝘪𝘢𝘵𝘪𝘰𝘯. 𝘈𝘴 𝘵𝘩𝘦 𝘏𝘶𝘭𝘬, 𝘩𝘦 𝘶𝘯𝘥𝘦𝘳𝘨𝘰𝘦𝘴 𝘢 𝘳𝘢𝘥𝘪𝘤𝘢𝘭 𝘵𝘳𝘢𝘯𝘴𝘧𝘰𝘳𝘮𝘢𝘵𝘪𝘰𝘯 𝘪𝘯𝘵𝘰 𝘢 𝘤𝘰𝘭𝘰𝘴𝘴𝘢𝘭, 𝘨𝘳𝘦𝘦𝘯-𝘴𝘬𝘪𝘯𝘯𝘦𝘥 𝘣𝘦𝘩𝘦𝘮𝘰𝘵𝘩 𝘸𝘪𝘵𝘩 𝘶𝘯𝘱𝘢𝘳𝘢𝘭𝘭𝘦𝘭𝘦𝘥 𝘴𝘵𝘳𝘦𝘯𝘨𝘵𝘩. 𝘛𝘩𝘦 𝘏𝘶𝘭𝘬 𝘴𝘦𝘳𝘷𝘦𝘴 𝘢𝘴 𝘢𝘯 𝘦𝘮𝘣𝘰𝘥𝘪𝘮𝘦𝘯𝘵 𝘰𝘧 𝘶𝘯𝘭𝘦𝘢𝘴𝘩𝘦𝘥 𝘱𝘰𝘸𝘦𝘳 𝘢𝘯𝘥 𝘵𝘩𝘦 𝘪𝘯𝘵𝘦𝘳𝘯𝘢𝘭 𝘴𝘵𝘳𝘶𝘨𝘨𝘭𝘦 𝘣𝘦𝘵𝘸𝘦𝘦𝘯 𝘤𝘰𝘯𝘵𝘳𝘰𝘭 𝘢𝘯𝘥 𝘤𝘩𝘢𝘰𝘴 𝘸𝘪𝘵𝘩𝘪𝘯 𝘉𝘢𝘯𝘯𝘦𝘳. 𝘛𝘩𝘪𝘴 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘥𝘦𝘭𝘷𝘦𝘴 𝘪𝘯𝘵𝘰 𝘵𝘩𝘦 𝘱𝘳𝘰𝘧𝘰𝘶𝘯𝘥 𝘪𝘮𝘱𝘢𝘤𝘵 𝘰𝘧 𝘴𝘤𝘪𝘦𝘯𝘵𝘪𝘧𝘪𝘤 𝘦𝘹𝘱𝘦𝘳𝘪𝘮𝘦𝘯𝘵𝘢𝘵𝘪𝘰𝘯 𝘰𝘯 𝘰𝘯𝘦'𝘴 𝘱𝘦𝘳𝘴𝘰𝘯𝘢, 𝘦𝘱𝘪𝘵𝘰𝘮𝘪𝘻𝘪𝘯𝘨 𝘵𝘩𝘦 𝘤𝘰𝘦𝘹𝘪𝘴𝘵𝘦𝘯𝘤𝘦 𝘰𝘧 𝘣𝘳𝘪𝘭𝘭𝘪𝘢𝘯𝘤𝘦 𝘢𝘯𝘥 𝘣𝘳𝘢𝘸𝘯. 𝘛𝘩𝘦 𝘏𝘶𝘭𝘬'𝘴 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘴𝘵𝘢𝘯𝘥𝘴 𝘢𝘴 𝘢 𝘵𝘦𝘴𝘵𝘢𝘮𝘦𝘯𝘵 𝘵𝘰 𝘵𝘩𝘦 𝘱𝘦𝘳𝘱𝘦𝘵𝘶𝘢𝘭 𝘤𝘰𝘯𝘧𝘭𝘪𝘤𝘵 𝘢𝘯𝘥 𝘴𝘺𝘮𝘣𝘪𝘰𝘴𝘪𝘴 𝘣𝘦𝘵𝘸𝘦𝘦𝘯 𝘪𝘯𝘵𝘦𝘭𝘭𝘦𝘤𝘵 𝘢𝘯𝘥 𝘣𝘳𝘶𝘵𝘦 𝘧𝘰𝘳𝘤𝘦 𝘸𝘪𝘵𝘩𝘪𝘯 𝘵𝘩𝘦 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘯𝘢𝘳𝘳𝘢𝘵𝘪𝘷𝘦.
  </div>
  
<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "HULK.jpg," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/ebce19df-60e5-408f-a8c0-8c549bf72f26">
</p>

---
  
# **CLINT BARTON AS HAWKEYE**![bow_and_arrow](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/d264b2cd-a9df-4d33-9edd-667f6a2182cf)

  <div align="justify">
𝘏𝘢𝘸𝘬𝘦𝘺𝘦, 𝘰𝘳 𝘊𝘭𝘪𝘯𝘵 𝘉𝘢𝘳𝘵𝘰𝘯, 𝘦𝘮𝘣𝘰𝘥𝘪𝘦𝘴 𝘢 𝘶𝘯𝘪𝘲𝘶𝘦 𝘦𝘹𝘱𝘭𝘰𝘳𝘢𝘵𝘪𝘰𝘯 𝘰𝘧 𝘥𝘶𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘪𝘦𝘴. 𝘐𝘯 𝘩𝘪𝘴 𝘳𝘦𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘊𝘭𝘪𝘯𝘵 𝘪𝘴 𝘢𝘯 𝘦𝘹𝘤𝘦𝘱𝘵𝘪𝘰𝘯𝘢𝘭𝘭𝘺 𝘴𝘬𝘪𝘭𝘭𝘦𝘥 𝘢𝘳𝘤𝘩𝘦𝘳 𝘢𝘯𝘥 𝘮𝘢𝘳𝘬𝘴𝘮𝘢𝘯, 𝘨𝘳𝘰𝘶𝘯𝘥𝘪𝘯𝘨 𝘵𝘩𝘦 𝘈𝘷𝘦𝘯𝘨𝘦𝘳𝘴 𝘸𝘪𝘵𝘩 𝘩𝘪𝘴 𝘩𝘶𝘮𝘢𝘯 𝘱𝘳𝘦𝘤𝘪𝘴𝘪𝘰𝘯 𝘢𝘮𝘪𝘥 𝘴𝘶𝘱𝘦𝘳𝘩𝘶𝘮𝘢𝘯 𝘢𝘣𝘪𝘭𝘪𝘵𝘪𝘦𝘴. 𝘈𝘴 𝘏𝘢𝘸𝘬𝘦𝘺𝘦, 𝘩𝘦 𝘣𝘦𝘤𝘰𝘮𝘦𝘴 𝘵𝘩𝘦 𝘶𝘯𝘴𝘶𝘯𝘨 𝘩𝘦𝘳𝘰 𝘸𝘪𝘦𝘭𝘥𝘪𝘯𝘨 𝘢 𝘣𝘰𝘸 𝘢𝘯𝘥 𝘢𝘳𝘳𝘰𝘸, 𝘴𝘩𝘰𝘸𝘤𝘢𝘴𝘪𝘯𝘨 𝘶𝘯𝘱𝘢𝘳𝘢𝘭𝘭𝘦𝘭𝘦𝘥 𝘢𝘤𝘤𝘶𝘳𝘢𝘤𝘺 𝘢𝘯𝘥 𝘢𝘥𝘢𝘱𝘵𝘢𝘣𝘪𝘭𝘪𝘵𝘺. 𝘊𝘭𝘪𝘯𝘵'𝘴 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘢𝘴 𝘢 𝘧𝘢𝘮𝘪𝘭𝘺 𝘮𝘢𝘯 𝘧𝘶𝘳𝘵𝘩𝘦𝘳 𝘢𝘥𝘥𝘴 𝘭𝘢𝘺𝘦𝘳𝘴 𝘵𝘰 𝘩𝘪𝘴 𝘤𝘩𝘢𝘳𝘢𝘤𝘵𝘦𝘳, 𝘦𝘮𝘱𝘩𝘢𝘴𝘪𝘻𝘪𝘯𝘨 𝘵𝘩𝘦 𝘳𝘦𝘭𝘢𝘵𝘢𝘣𝘭𝘦 𝘢𝘴𝘱𝘦𝘤𝘵𝘴 𝘰𝘧 𝘩𝘪𝘴 𝘭𝘪𝘧𝘦 𝘣𝘦𝘺𝘰𝘯𝘥 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘥𝘶𝘵𝘪𝘦𝘴. 𝘛𝘩𝘦 𝘴𝘪𝘮𝘱𝘭𝘪𝘤𝘪𝘵𝘺 𝘰𝘧 𝘢 𝘣𝘰𝘸 𝘤𝘰𝘯𝘵𝘳𝘢𝘴𝘵𝘴 𝘸𝘪𝘵𝘩 𝘵𝘩𝘦 𝘨𝘳𝘢𝘯𝘥𝘦𝘶𝘳 𝘰𝘧 𝘴𝘶𝘱𝘦𝘳𝘱𝘰𝘸𝘦𝘳𝘴, 𝘺𝘦𝘵 𝘏𝘢𝘸𝘬𝘦𝘺𝘦'𝘴 𝘳𝘦𝘴𝘪𝘭𝘪𝘦𝘯𝘤𝘦 𝘢𝘯𝘥 𝘴𝘵𝘳𝘢𝘵𝘦𝘨𝘪𝘤 𝘣𝘳𝘪𝘭𝘭𝘪𝘢𝘯𝘤𝘦 𝘮𝘢𝘬𝘦 𝘩𝘪𝘮 𝘢𝘯 𝘪𝘯𝘥𝘪𝘴𝘱𝘦𝘯𝘴𝘢𝘣𝘭𝘦 𝘈𝘷𝘦𝘯𝘨𝘦𝘳. 𝘊𝘭𝘪𝘯𝘵 𝘉𝘢𝘳𝘵𝘰𝘯'𝘴 𝘤𝘩𝘢𝘳𝘢𝘤𝘵𝘦𝘳 𝘶𝘯𝘥𝘦𝘳𝘴𝘤𝘰𝘳𝘦𝘴 𝘵𝘩𝘢𝘵 𝘦𝘷𝘦𝘯 𝘪𝘯 𝘢 𝘸𝘰𝘳𝘭𝘥 𝘰𝘧 𝘨𝘰𝘥𝘴 𝘢𝘯𝘥 𝘮𝘰𝘯𝘴𝘵𝘦𝘳𝘴, 𝘢 𝘮𝘰𝘳𝘵𝘢𝘭 𝘸𝘪𝘵𝘩 𝘶𝘯𝘸𝘢𝘷𝘦𝘳𝘪𝘯𝘨 𝘴𝘬𝘪𝘭𝘭 𝘢𝘯𝘥 𝘥𝘦𝘵𝘦𝘳𝘮𝘪𝘯𝘢𝘵𝘪𝘰𝘯 𝘤𝘢𝘯 𝘤𝘢𝘳𝘷𝘦 𝘢 𝘴𝘪𝘨𝘯𝘪𝘧𝘪𝘤𝘢𝘯𝘵 𝘴𝘱𝘢𝘤𝘦 𝘸𝘪𝘵𝘩𝘪𝘯 𝘵𝘩𝘦 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘯𝘢𝘳𝘳𝘢𝘵𝘪𝘷𝘦.
  </div>

<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "HAWKEYE.png," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/2e9a28fb-b457-4a18-9d69-3e5c6535fa2f">
</p>

---
  
# **NATASHA ROMANOFF AS BLACK WIDOW**![blackwidow](https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/226a6c79-7223-4c5c-afc5-bbf77cfae4d5)

  <div align="justify">
𝘉𝘭𝘢𝘤𝘬 𝘞𝘪𝘥𝘰𝘸, 𝘢𝘭𝘴𝘰 𝘬𝘯𝘰𝘸𝘯 𝘢𝘴 𝘕𝘢𝘵𝘢𝘴𝘩𝘢 𝘙𝘰𝘮𝘢𝘯𝘰𝘧𝘧, 𝘯𝘢𝘷𝘪𝘨𝘢𝘵𝘦𝘴 𝘢 𝘤𝘰𝘮𝘱𝘦𝘭𝘭𝘪𝘯𝘨 𝘥𝘪𝘤𝘩𝘰𝘵𝘰𝘮𝘺 𝘰𝘧 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘪𝘦𝘴. 𝘐𝘯 𝘩𝘦𝘳 𝘳𝘦𝘢𝘭 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺, 𝘴𝘩𝘦 𝘪𝘴 𝘢 𝘴𝘬𝘪𝘭𝘭𝘦𝘥 𝘴𝘱𝘺 𝘢𝘯𝘥 𝘧𝘰𝘳𝘮𝘦𝘳 𝘢𝘴𝘴𝘢𝘴𝘴𝘪𝘯, 𝘴𝘩𝘳𝘰𝘶𝘥𝘦𝘥 𝘪𝘯 𝘢 𝘮𝘺𝘴𝘵𝘦𝘳𝘪𝘰𝘶𝘴 𝘱𝘢𝘴𝘵. 𝘈𝘴 𝘉𝘭𝘢𝘤𝘬 𝘞𝘪𝘥𝘰𝘸, 𝘴𝘩𝘦 𝘵𝘳𝘢𝘯𝘴𝘧𝘰𝘳𝘮𝘴 𝘪𝘯𝘵𝘰 𝘢 𝘧𝘰𝘳𝘮𝘪𝘥𝘢𝘣𝘭𝘦 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰, 𝘴𝘩𝘰𝘸𝘤𝘢𝘴𝘪𝘯𝘨 𝘶𝘯𝘱𝘢𝘳𝘢𝘭𝘭𝘦𝘭𝘦𝘥 𝘤𝘰𝘮𝘣𝘢𝘵 𝘴𝘬𝘪𝘭𝘭𝘴 𝘢𝘯𝘥 𝘴𝘵𝘳𝘢𝘵𝘦𝘨𝘪𝘤 𝘱𝘳𝘰𝘸𝘦𝘴𝘴. 𝘕𝘢𝘵𝘢𝘴𝘩𝘢'𝘴 𝘫𝘰𝘶𝘳𝘯𝘦𝘺 𝘪𝘯𝘷𝘰𝘭𝘷𝘦𝘴 𝘳𝘦𝘤𝘰𝘯𝘤𝘪𝘭𝘪𝘯𝘨 𝘩𝘦𝘳 𝘤𝘰𝘷𝘦𝘳𝘵 𝘰𝘱𝘦𝘳𝘢𝘵𝘪𝘷𝘦 𝘩𝘪𝘴𝘵𝘰𝘳𝘺 𝘸𝘪𝘵𝘩 𝘩𝘦𝘳 𝘯𝘦𝘸𝘧𝘰𝘶𝘯𝘥 𝘢𝘭𝘭𝘦𝘨𝘪𝘢𝘯𝘤𝘦 𝘵𝘰 𝘵𝘩𝘦 𝘈𝘷𝘦𝘯𝘨𝘦𝘳𝘴. 𝘛𝘩𝘦 𝘳𝘦𝘥 𝘪𝘯 𝘩𝘦𝘳 𝘭𝘦𝘥𝘨𝘦𝘳 𝘴𝘦𝘳𝘷𝘦𝘴 𝘢𝘴 𝘢 𝘤𝘰𝘯𝘴𝘵𝘢𝘯𝘵 𝘳𝘦𝘮𝘪𝘯𝘥𝘦𝘳 𝘰𝘧 𝘵𝘩𝘦 𝘴𝘩𝘢𝘥𝘰𝘸𝘴 𝘴𝘩𝘦 𝘴𝘦𝘦𝘬𝘴 𝘵𝘰 𝘰𝘷𝘦𝘳𝘤𝘰𝘮𝘦. 𝘛𝘩𝘦 𝘤𝘩𝘢𝘳𝘢𝘤𝘵𝘦𝘳 𝘰𝘧 𝘉𝘭𝘢𝘤𝘬 𝘞𝘪𝘥𝘰𝘸 𝘦𝘹𝘦𝘮𝘱𝘭𝘪𝘧𝘪𝘦𝘴 𝘵𝘩𝘦 𝘪𝘯𝘵𝘳𝘪𝘤𝘢𝘵𝘦 𝘥𝘢𝘯𝘤𝘦 𝘣𝘦𝘵𝘸𝘦𝘦𝘯 𝘵𝘩𝘦 𝘦𝘭𝘶𝘴𝘪𝘷𝘦 𝘯𝘢𝘵𝘶𝘳𝘦 𝘰𝘧 𝘢 𝘴𝘱𝘺'𝘴 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘢𝘯𝘥 𝘵𝘩𝘦 𝘪𝘯𝘥𝘰𝘮𝘪𝘵𝘢𝘣𝘭𝘦 𝘴𝘱𝘪𝘳𝘪𝘵 𝘰𝘧 𝘢 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰. 𝘏𝘦𝘳 𝘦𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯 𝘶𝘯𝘥𝘦𝘳𝘴𝘤𝘰𝘳𝘦𝘴 𝘵𝘩𝘦 𝘳𝘦𝘴𝘪𝘭𝘪𝘦𝘯𝘤𝘦 𝘳𝘦𝘲𝘶𝘪𝘳𝘦𝘥 𝘵𝘰 𝘳𝘦𝘥𝘦𝘧𝘪𝘯𝘦 𝘰𝘯𝘦𝘴𝘦𝘭𝘧 𝘣𝘦𝘺𝘰𝘯𝘥 𝘢 𝘮𝘶𝘳𝘬𝘺 𝘱𝘢𝘴𝘵, 𝘴𝘩𝘰𝘸𝘤𝘢𝘴𝘪𝘯𝘨 𝘵𝘩𝘦 𝘤𝘰𝘮𝘱𝘭𝘦𝘹𝘪𝘵𝘺 𝘰𝘧 𝘪𝘥𝘦𝘯𝘵𝘪𝘵𝘺 𝘸𝘪𝘵𝘩𝘪𝘯 𝘵𝘩𝘦 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘯𝘢𝘳𝘳𝘢𝘵𝘪𝘷𝘦. 
    </div>

<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "BLACKWIDOW.png," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/ClarenceJoyIlagan/Group8_Finals_FaceRecognition/assets/144084976/a4d68a6d-5d4d-49a0-b36e-43416a2d2423">
</p>

---

<div align="center">
  
### **𝗦𝗼𝗺𝗲 𝗼𝗳 𝘁𝗵𝗲 𝗖𝗮𝘀𝘁𝘀 𝗮𝗹𝗼𝗻𝗴𝘀𝗶𝗱𝗲 𝘁𝗵𝗲 𝗢𝗿𝗶𝗴𝗶𝗻𝗮𝗹 𝗦𝗶𝘅**
### <sup>🇺🇳🇰🇳🇴🇼🇳 🇽 🇫🇦🇨🇪 🇷🇪🇨🇴🇬🇳🇮🇹🇹🇮🇴🇳</sup>

</div>

<div align="center">
  
_In the ever-expanding universe of MCU, a diverse array of heroes has risen to prominence, each leaving an indelible mark on the epic saga that began with the Avengers Original Six. As the MCU evolved through its phases, a new generation of extraordinary individuals emerged, contributing their unique powers, stories, and dynamics to the grand narrative. These heroes, with their diverse abilities and compelling narratives, seamlessly integrate into the MCU's larger-than-life tableau. As we navigate through these individuals, we relive a cinematic tapestry such as uniting heroes across time, space, and the shared pursuit of a universe protected by the rallying cry: "Avengers, assemble!"_

</div>

---

# **CAST 1**

<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "A1.png," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/aprilrhose/Group8_Finals_FaceRecognition/assets/143881769/bee0139d-b1eb-4480-8b52-c2df960c8f54">
</p>

---

# **CAST 2**

<br>

_<sup>🔻Using the provided face recognition code and analyzing the image "A2.png," the following results were obtained:🔻</sup>_

<br>

<p align="center">
  <img src="https://github.com/aprilrhose/Group8_Finals_FaceRecognition/assets/143881769/374d76bc-eeb5-428f-8b30-a20b7b34ccdd">
</p>

---


