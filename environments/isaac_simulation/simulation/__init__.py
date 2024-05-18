
from .need_cup import NeedCup
from .need_control import NeedControl
from .need_charger import NeedCharger
from .need_glasses import NeedGlasses
from .need_rubick import NeedRubick
from .need_cutter import NeedCutter
from .feel_hungry import FeelHungry
from .feel_distracted import FeelDistracted

from .select_expensive import SelectExpensive
from .select_vitaminc import SelectVitaminc
from .select_in_library import SelectInLibrary

from .need_phone_under_the_cup import NeedPhoneUnderCup

task_dic={
"need_cup":NeedCup,
"need_control":NeedControl,
"need_charger":NeedCharger,
"need_glasses":NeedGlasses,
"need_rubick":NeedRubick,

"need_cutter":NeedCutter,

"feel_hungry":FeelHungry,
"feel_distracted":FeelDistracted,

"select_expensive":SelectExpensive,
"select_vitaminc":SelectVitaminc,
"select_in_library":SelectInLibrary,


"need_phone_under_cup":NeedPhoneUnderCup,

}