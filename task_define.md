Task1:
    Prompt:
        I plan to turn on the TV, put what I need {position_object} the {position_object}. 
    Answer:
        (remote controller,_)

Task2:
    Prompt:
        I plan to use my iphone but my eyes are bad, put what I need {position_object} the iphone.
    Answer:
        (glasses,_)

Task3:
    Prompt:
        I plan to relax my fingers, put what I need {position_object} the {position_object}.
    Answer:
        (rubick,_)

Task4:
    Prompt:
        I plan to charge my phone, put what I need {position_object} the phone.
    Answer:
        (charger,_)

Task5:
    Prompt:
        I plan to unwarp the gift, put what I need {position_object} the gift.
    
    Answer:
        (cutter,_)

Task6:
    Prompt:
        I am feeling thirsty, put what I need {position_object} the {position_object}.
    Answer:
        (cup,_)

Task7:
    Prompt:
        I am feeling hungry, put what I need {position_object} the {position_object}.
    Answer:
        (cupcake,_)

Task8:
    Prompt:
        I am feeling distracted. The environment is too noisy, Put what I need {position_object} the {position_object}.
    Answer:
        (headphones,_)

Task9:
    Prompt:
        Put {key of feature_dict} {position_word} the {position_object}.
    Answer:
        (value of feature_dict,_)


Task10:
    Prompt:
        Place {object_under} {position_word} the {position_object}. Be careful the excution order.
    Answer:
        (object_above,_)
        (object_under,_)

``` python

实际上这个 "cup" -> ("cup",urdf_path)
position_dict:
{
"position_object":["cup","laptop","phone","gift","red bowl","box","black area"],   
"position_word":[ "in front of","behind","on the left of","on the right of"]
}

 
# 选择的时候，干扰项就不选这个位置里面的所有词
feature_dict:
{ "most expensive object":["laptop","phone"],
"object which contains the vitamin C":["orange","lemon"],
"object which might appear in the library":["book","pen"],
"object which might appear in the kitchen":["knife","fork"]
}


# 主要考虑能不能放稳，不会掉下来
under_above_dict:
{
"object_under":["phone","book","cup","box"],
"object_above":["cup","book","phone","box","black maker","red bowl"]
}

```



draft 还没想好怎么归档，先放这里


5. Sort the table {} {}
    
6. put the unrelated object into the box
  - {to use my laptop but the screen is dirty} {, put what I need in front of the laptop} 
- - {to charge my laptop} {, put what I need in front of the laptop}
