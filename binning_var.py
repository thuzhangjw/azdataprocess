class ContinuousVariables(object):
    def __init__(self, item_name, normal_value_list):
        self.name = item_name
        self.normal_value_list = normal_value_list


ABG = ContinuousVariables('(ABG)白蛋白比球蛋白', [['', 1.2, 2.4]])
ALB = ContinuousVariables('(ALB)白蛋白', [['g/dl', 3.5, 5.3], ['g/L',40,55], ['g/l',40,55]]) # 补全
ALP = ContinuousVariables('(ALP)碱性磷酸酶', [['U/L', 42, 98]]) # 照片上有冲突 有些['U/L',45, 125]
ALT = ContinuousVariables('(ALT)谷丙转氨酶', [['U/L', 0, 42]])
ANTIHBC = ContinuousVariables('(ANTIHBC)乙肝核心抗体', [['S/CO', 0, 1]])
ANTIHBE = ContinuousVariables('(ANTIHBE)乙肝e抗体', [['S/CO', 1, 9999]])
ANTIHBS = ContinuousVariables('(ANTIHBS)乙肝表面抗体', [['mIU/ml', 0, 10], ['mIU/mL', 0, 10]])
ANTIHCV = ContinuousVariables('(ANTIHCV)丙肝抗体-IgG', [['S/CO', 0, 1]])
# ANTIHIV = ContinuousVariables('(ANTIHIV)艾滋病毒抗体', [['', , ]]) # 没找到
APTT = ContinuousVariables('(APTT)活化部分凝血活酶时间', [['Sec', 25.1, 36.5], ['S', 25.1, 36.5]])
AST = ContinuousVariables('(AST)谷草转氨酶', [['U/L', 8, 40]])
BAS = ContinuousVariables('(BAS)嗜碱细胞数', [['G/L', 0.02, 0.05], ['10～9/L', 0.02, 0.05]]) #补全
BASBFB = ContinuousVariables('(BASBFB)嗜碱细胞百分比', [['%', 0, 1]]) # 需核查
BLD = ContinuousVariables('(BLD)潜血/隐血', [['g/L', 120, 160]]) # 需核查
BUN = ContinuousVariables('(BUN)尿素', [['mmol/L', 2.8, 7.2], ['mg/dl',9,20]]) # 补全
CHE = ContinuousVariables('(CHE)胆碱酯酶', [['kU/L', 4.62, 11.5], ['KU/L', 4.62, 11.5]])
CK = ContinuousVariables('(CK)磷酸肌酸激酶', [['U/L', 24, 195]])
CKMB = ContinuousVariables('(CKMB)肌酸激酶同工酶', [['ng/ml',0,3.6], ['U/L', 0, 25], ['ug/L',0,3.6]]) # 补全
DBIL = ContinuousVariables('(DBIL)直接胆红素', [['umol/L', 0.5, 6.8], ['μmol/L',0.5,6.8], ['mg/dl',0.1,0.4]]) # 补全
# DDIMER = ContinuousVariables('(DDIMER)D-二聚体定量', [['μg/l',,], ['ug/L',,], ['ng/ml',,]]) # 补全
EOS = ContinuousVariables('(EOS)嗜酸细胞数', [['G/L', 0.05, 0.3], ['10～9/L',0.05,0.3], ['/MM',50,300]])
EOSBFB = ContinuousVariables('(EOSBFB)嗜酸细胞百分比', [['%', 0.5, 5]])
FBG = ContinuousVariables('(FBG)纤维蛋白原定量', [['g/l', 2, 4], ['mg/dl',200,400], ['g/L', 2, 4]])
GA = ContinuousVariables('(GA)糖化血清白蛋白', [['%',11,16]]) # 实际无g/L
GLU = ContinuousVariables('(GLU)葡萄糖', [['mg/dL', 70, 110], ['mg/dl', 70, 110], ['mmol/L',3.9,7.8], ['m mol/L',3.9,7.8]])
HB = ContinuousVariables('(HB)血红蛋白', [['g/L', 120, 160], ['g/l', 120, 160], ['G/L', 120, 160], ['g/dL',12,16], ['g/dl',12,16]])
HBEAG = ContinuousVariables('(HBEAG)乙肝e抗原', [['S/CO', 0, 1]])
HBSAG = ContinuousVariables('(HBSAG)乙肝表面抗原', [['IU/ml', 0, 0.05], ['IU/mL', 0, 0.05]])
# HBSAG = ContinuousVariables('(HBSAG)乙肝表面抗原', [['IU/ml', 0, 0.05], ['IU/mL', 0, 0.05], ['S/N',,]])
HCT = ContinuousVariables('(HCT)*红细胞压积', [['L/L', 0.4, 0.5], ['%', 40, 50]])
HDL = ContinuousVariables('(HDL)高密度脂蛋白', [['mg/dl', 29.8, 87],['mmol/L',0.9,1.83]])
HSCRP = ContinuousVariables('(HSCRP)C-反应蛋白(高敏/超敏)', [['mg/l', 0, 5]])
INR = ContinuousVariables('(INR)国际标准化比值', [['', 0.8, 1.2]])
KET = ContinuousVariables('(KET)尿酮体', [['g/L', 120, 160]])
LDH = ContinuousVariables('(LDH)乳酸脱氢酶', [['U/L', 135, 225]]) # 根据检测方法不一
LDL = ContinuousVariables('(LDL)低密度脂蛋白', [['mg/dl', 49.1, 159.6], ['mmol/L',0,3.12]])
LYM = ContinuousVariables('(LYM)淋巴细胞数', [['G/L', 1.5, 4.5], ['10～9/L',1.5, 4.5]])
LYMBFB = ContinuousVariables('(LYMBFB)淋巴细胞百分比', [['%', 20, 40]]) # 有多个不同的正常值
MCH = ContinuousVariables('(MCH)平均红细胞Hb含量', [['g/L', 120, 160], ['pg',26,38]]) # g/L 的单位没找到
MCHC = ContinuousVariables('(MCHC)平均红细胞Hb浓度', [['g/L', 300, 360], ['G/L',300,360]])
MCV = ContinuousVariables('(MCV)*红细胞平均体积', [['fL', 82.0, 92.0], ['fl', 82.0, 92.0]])
MON = ContinuousVariables('(MON)单核细胞数', [['G/L', 0.2, 0.8]])
MONBFB = ContinuousVariables('(MONBFB)单核细胞百分比', [['%', 50, 75]])
MPV = ContinuousVariables('(MPV)平均血小板体积', [['fL', 6.8, 13.5], ['fl', 6.8, 13.5]])
NE = ContinuousVariables('(NE)中性粒细胞数', [['G/L', 2.0, 7.5], ['10～9/L', 2.0, 7.5]])
NEBFB = ContinuousVariables('(NEBFB)中性粒细胞百分比', [['%', 15.5, 18.1]])
# NIT = ContinuousVariables('(NIT)亚硝酸盐', [['', , ]]) # 没找到
# PA = ContinuousVariables('(PA)梅毒螺旋体抗体确证实验', [['S/CO', , ]])
PALB = ContinuousVariables('(PALB)前白蛋白', [['g/L', 0.18, 0.41], ['mg/L',180,410]])
PCT = ContinuousVariables('(PCT)血小板压积', [['%', 0.11, 0.28], ['mL/L', 0.11, 0.28]]) # 单位换算似乎不对
PDW = ContinuousVariables('(PDW)血小板体积分布宽度', [['%', 15.5,18.1], ['10(GSD)', 15.5,18.1]])
PH = ContinuousVariables('(PH)酸碱度', [['', 7.35, 7.45]])
PLCR = ContinuousVariables('(PLCR)大血小板比率', [['%', 10, 16]])

AG    = ContinuousVariables('阴离子间隙',                 [['mmol/L',8,16]]) #!无参考范围？？？！！！
P     = ContinuousVariables('无机磷测定(P)',              [['mg/dl',2.5,4.8], ['mmol/L',0.74,1.2]])
MG    = ContinuousVariables('镁(MG)',                    [['mmol/L',0.8,1.2]])
CL    = ContinuousVariables('氯(CL)',                    [['mmol/L',97,110]])
K     = ContinuousVariables('钾(K)',                     [['mmol/L',3.5,5.3]])
MB    = ContinuousVariables('肌酸激酶同功酶MB',            [['ug/L',0,3.6], ['U/L',0,25], ['ng/ml',0,3.6]])
CA    = ContinuousVariables('钙(CA)',                    [['mg/dl',8,10], ['mmol/L',1.09,1.3]])
I     = ContinuousVariables('超敏肌钙蛋白I',               [['ng/ml',17.4,105.7]]) # 检验单上写的是['ng/ml', 0, 0.04]
# pending BOBB  = ContinuousVariables('便潜血（BOBB）')
WBC   = ContinuousVariables('(WBC)白细胞',                [['10～9/L',4,10], ['G/L',4,10]])
# UWBC  = ContinuousVariables('(UWBC)尿白细胞',             [['/ul',,], ['uL',,], ['/μl',,]]) #没找到
# URBC  = ContinuousVariables('(URBC)尿红细胞',             [['/ul',,], ['uL',,], ['/高倍视野',,], ['/μl',,]]) #没找到
UA    = ContinuousVariables('(UA)尿酸',                   [['μmol/L',208.3,428], ['mg/dl',3.4,7], ['umol/L',208.3,428]]) #没找全
TP    = ContinuousVariables('(TP)总蛋白',                 [['g/dl',6.2,8.3], ['g/L',62,83]])
TNI   = ContinuousVariables('(TNI)肌钙蛋白',              [['ng/ml',0,0.15]]) # 没找到
TG    = ContinuousVariables('(TG)甘油三酯',               [['mg/dl',0,150], ['mmol/L',0,1.7]]) #没找全
TCO2  = ContinuousVariables('(TCO2)总CO2',                [['mmol/L',21,31.3]])
TC    = ContinuousVariables('(TC)总胆固醇',               [['mh/dl',0,200], ['mmol/L',3,5.2], ['mmol/l',3,5.2], ['mg/dl',120,221]]) #没找全
TBIL  = ContinuousVariables('(TBIL)总胆红素',              [['μmol/L',5,21], ['mg/dl',0.2,1.2], ['umol/L',5,21]]) #没找全
TBA   = ContinuousVariables('(TBA)血清总胆汁酸',           [['μmol/L',0,12], ['umol/l',0,12], ['umol/L',0,12]])
SG    = ContinuousVariables('(SG)比重',                   [['',1.002,1.03]])
RGT   = ContinuousVariables('(RGT)γ谷氨酰转肽酶',          [['U/L',3,50]]) # 照片中有冲突，有图片表示 ['U/L',10,60]
RDWSD = ContinuousVariables('(RDWSD)红细胞分布宽度-SD值',   [['fl',11,16]]) # 没找到！
RDWCV = ContinuousVariables('(RDWCV)红细胞分布宽度-CV值',   [['%',11.6,14.8]])
RBC   = ContinuousVariables('(RBC)红细胞计数',             [['10～12/L',4.0,5.5], ['T/L',4.0,5.5]])
PT    = ContinuousVariables('(PT)凝血酶原时间',             [['Sec',9.4,12.5], ['S',9.4,12.5], ['s',9.4,12.5]])
PLT   = ContinuousVariables('(PLT)血小板',                 [['10～9/L',100,300], ['G/L',100,300]])


_lab_tests = [ABG, ALB, ALP, ALT, ANTIHBC, ANTIHBE, ANTIHBS, ANTIHCV, APTT, AST, BAS, BASBFB, BLD, BUN, CHE, CK, CKMB, DBIL,
             EOS, EOSBFB, FBG, GA, GLU, HB, HBEAG, HBSAG, HCT, HDL, HSCRP, INR, KET, LDH, LDL, LYM, LYMBFB, MCH, MCHC, MCV,
             MON, MONBFB, MPV, NE, NEBFB, PALB, PCT, PDW, PH, PLCR, AG, P, MG, CL, K, MB, CA, I, WBC, UA, TP, TNI, TG, TCO2,
             TC, TBIL, TBA, SG, RGT, RDWSD, RDWCV, RBC, PT, PLT]


HEART_RATE = ContinuousVariables('心率', [['', 60, 100]])
DIASTOLIC = ContinuousVariables('舒张压', [['', 60, 90]])
SYSTOLIC = ContinuousVariables('收缩压', [['', 90, 139]])
TEMPER = ContinuousVariables('体温', [['', 36, 37]])
BREATH = ContinuousVariables('呼吸', [['', 16, 20]])

_others = [HEART_RATE, DIASTOLIC, SYSTOLIC, TEMPER, BREATH]

continuous_vars = _lab_tests + _others


