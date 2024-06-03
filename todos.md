# todo

## grasp related

1. 如何实现抓取物体路径，具体的hand pose

2. urdf定义某些奇怪物体的collision box


## isaac related

1. 有些东西需要手动旋转一下，config文件预计需要补充一些旋转的信息

2. 随机生成位置大概率需要get到已有物体的大小，但是实际上物体obj到urdf我在config里面有一个缩放（因为gym缩放会有bug），需要一个桥梁获得到obj scaled 之后的大小（通过一些脚本应该可以做到）

3. 物品质量设置后的模拟让一些物体感觉不是很自然的放在桌子上，不知道引擎模拟有什么问题


## task related

目前基本范式实现的认为是ok的，可能有些软工级别的修复需要调整一下

想起来执行顺序的检测应该加一个第一个物体 todo
