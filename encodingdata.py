import pandas as pd


class DiagnoseData(object):
    log = open('logs.txt', 'w')
    instancenum = 0

    def __init__(self, columnname, datalist):
        self.columnname = columnname
        self.datalist = datalist
        DiagnoseData.instancenum += 1
        print(self.columnname + '  创建, instance num = ' + str(DiagnoseData.instancenum))

    def __del__(self):
        DiagnoseData.instancenum -= 1
        print(self.columnname + '  析构, instance num = ' + str(DiagnoseData.instancenum))
        if DiagnoseData.instancenum == 0:
            DiagnoseData.log.close()
            print('日志文件关闭')


class EnumData(DiagnoseData):
    def __init__(self, columnname, datalist, enums):
        super(EnumData, self).__init__(columnname, datalist)
        self.enums = enums

    def encoding2onehot(self):
        m = {}
        for i in range(len(self.enums)):
            m[self.columnname + '_' + self.enums[i]] = [0] * len(self.datalist)
        res = pd.DataFrame(m)
        for idx, val in enumerate(self.datalist):
            flag = True
            for enm in self.enums:
                if val == enm:
                    flag = False
                    res.loc[idx, self.columnname + '_' + enm] = 1
                    break
            if flag and pd.notna(val):
                DiagnoseData.log.write(self.columnname + ': line ' + str(idx) + ' : ' + str(val) + '\n')
        return res


class BlockData(DiagnoseData):
    def __init__(self, columnname, datalist, blocks):
        super(BlockData, self).__init__(columnname, datalist)
        self.blocks = blocks

    def encoding2onehot(self):
        m = {}
        for i in range(len(self.blocks)):
            m[self.columnname + '_' + str(i)] = [0] * len(self.datalist)
        res = pd.DataFrame(m)
        for idx, val in enumerate(self.datalist):
            flag = True
            for blcidx, blc in enumerate(self.blocks):
                if blc[0] <= val < blc[1]:
                    flag = False
                    res.loc[idx, self.columnname + '_' + str(blcidx)] = 1
                    break
            if flag and pd.notna(val):
                DiagnoseData.log.write(self.columnname + ': line ' + str(idx) + ' : ' + str(val) + '\n')
        return res


df = pd.read_excel('../data/finalNoEncoding.xlsx', dtype=object)
pendingjobs = [
    EnumData('性别', df['性别'], ['男', '女']),
    BlockData('年龄', df['年龄'], [(0, 6), (6, 18), (18, 28), (28, 40), (40, 65), (65, 75), (75, 110)]),
    # EnumData('民族', df['民族'], [...]),
    EnumData('血型', df['血型'], ['A', 'B', 'AB', 'O']),
    BlockData('舒张压', df['舒张压'], [(0, 60), (60, 90), (90, 120), (120, 200)]),
    BlockData('收缩压', df['收缩压'], [(0, 90), (90, 140), (140, 200), (200, 300)]),
    EnumData('心脏心律', df['心脏心律'], ['绝对不齐', '不齐', '齐']),
    EnumData('心音s1', df['心音s1'], ['分裂', '减弱', '强弱不等', '消失', '增强', '正常']),
    EnumData('心音s2', df['心音s2'], ['分裂', '减弱', '强弱不等', '消失', '增强', '正常']),
    EnumData('心音s3', df['心音s3'], ['无', '有']),
    EnumData('心音s4', df['心音s4'], ['无', '有']),
    EnumData('心音a2', df['心音a2'], ['减弱', '亢进', '正常']),
    EnumData('心音p2', df['心音p2'], ['减弱', '亢进', '正常']),
    EnumData('心音a2和p2关系', df['心音a2和p2关系'], ['>|大于', '=|等于', '<|小于']),
    BlockData('心率', df['心率'], [(0, 50), (50, 60), (60, 100), (100, 200), (200, 300)]),
    BlockData('(WBC)白细胞', df['(WBC)白细胞_G/L'], [(0, 3.2), (3.2, 4), (4, 10.01), (10.01, 12), (12, 20), (20, 100)]),
    BlockData('(PLT)血小板', df['(PLT)血小板_G/L'], [(0, 80), (80, 100), (100, 301), (301, 360), (360, 600), (600, 900)]),
    BlockData('(RBC)红细胞计数', df['(RBC)红细胞计数_T/L'], [(0, 3.2), (3.2, 4), (4, 5.51), (5.51, 6.6), (6.6, 10)]),
    BlockData('(MCHC)平均红细胞Hb浓度', df['(MCHC)平均红细胞Hb浓度_g/L'], [(0, 240), (240, 300), (300, 360), (360, 430), (430, 720)]),
    BlockData('(HB)血红蛋白', df['(HB)血红蛋白_g/dl'], [(0, 8.8), (8.8, 11), (11, 16.1), (16.1, 19.2), (19.2, 32)]),
    BlockData('(LYMBFB)淋巴细胞百分比', df['(LYMBFB)淋巴细胞百分比_%'], [(0, 16), (16, 20), (20, 40.1), (40.1, 48), (48, 80), (80, 100)]),
    BlockData('(MONBFB)单核细胞百分比', df['(MONBFB)单核细胞百分比_%'], [(0, 2.4), (2.4, 3), (3, 8.1), (8.1, 9.6), (9.6, 16), (16, 24)]),
    BlockData('(NE)中性粒细胞数', df['(NE)中性粒细胞数_G/L'], [(0, 1.44), (1.44, 1.8), (1.8, 6.3), (6.3, 7.56), (7.56, 12.6)]),
    BlockData('(MON)单核细胞数', df['(MON)单核细胞数_G/L'], [(0, 0.24), (0.24, 0.3), (0.3, 0.81), (0.81, 0.96), (0.96, 1.6), (1.6, 3.2)]),
    BlockData('(HCT)红细胞压积', df['(HCT)红细胞压积_%'], [(0, 28), (28, 35), (35, 50), (50, 60), (60, 80)]),
    BlockData('(MCV)红细胞平均体积', df['(MCV)红细胞平均体积_fL'], [(0, 64), (64, 80), (80, 100.1), (100.1, 125)]),
    BlockData('(RDWSD)红细胞分布宽度-SD值', df['(RDWSD)红细胞分布宽度-SD值_fl'], [(0, 28), (28, 35), (35, 56.1), (56.1, 67.3), (67.3, 90)]),
    BlockData('(RDWCV)红细胞分布宽度-CV值', df['(RDWCV)红细胞分布宽度-CV值_%'], [(11, 16), (16, 19.3), (19.3, 32)]),
    BlockData('(MPV)平均血小板体积', df['(MPV)平均血小板体积_fL'], [(0, 5.6), (5.6, 7), (7, 13.1), (13.1, 15.7)]),
    BlockData('(EOSBFB)嗜酸细胞百分比', df['(EOSBFB)嗜酸细胞百分比_%'], [(0, 0.32), (0.32, 0.4), (0.4, 8.01), (8.01, 9.61), (9.61, 16), (16, 34)]),
    BlockData('(BAS)嗜碱细胞数', df['(BAS)嗜碱细胞数_G/L'], [(0, 0.2), (0.2, 0.24), (0.24, 0.4), (0.4, 1), (1, 2)]),
    BlockData('(PLCR)大血小板比率', df['(PLCR)大血小板比率_%'], [(0, 10), (10, 13), (13, 45), (45, 54), (54, 80)]),
    BlockData('(BUN)尿素', df['(BUN)尿素_mmol/L'], [(0, 2.24), (2.24, 2.8), (2.8, 7.21), (7.21, 8.64), (8.64, 14.4), (14.4, 28.8), (28.8, 50)]),
    BlockData('(CREA)肌酐', df['(CREA)肌酐_mmol/L'], [(0, 0.0352), (0.0352, 0.044), (0.044, 0.1061), (0.1061, 0.1272), (0.1272, 0.212), (0.212, 0.5)]),
    BlockData('(HDL)高密度脂蛋白', df['(HDL)高密度脂蛋白_mmol/L'], [(0, 0.56), (0.56, 0.7), (0.7, 2.01), (2.01, 2.4), (2.4, 4)]),  # 0.7--2.0
    BlockData('(LDL)低密度脂蛋白', df['(LDL)低密度脂蛋白_mmol/L'], [(0, 3.121), (3.121, 3.744), (3.744, 6.241), (6.241, 12.481)]),  # 0--3.12
    BlockData('(TC)总胆固醇', df['(TC)总胆固醇_mmol/L'], [(0, 2.281), (2.281, 5.7), (5.7, 6.84), (6.84, 11.4), (11.4, 22.8)]),  # 2.85--5.69
    BlockData('(TG)甘油三酯', df['(TG)甘油三酯_mmol/L'], [(0, 0.36), (0.36, 0.45), (0.45, 1.7), (1.7, 2.251), (2.251, 6.8), (6.8, 15), (15, 50)]),  #  0.45--1.69
    BlockData('(UA)尿酸', df['(UA)尿酸_umol/L'], [(0, 71.2), (71.2, 89), (89, 416.1), (416.1, 499.2), (499.2, 832), (832, 1200)]),  # 89--416
    # BlockData('(HSCRP)C-反应蛋白(高敏/超敏)_mg/l'),
    BlockData('钾(K)', df['钾(K)_mmol/L'], [(0, 2.8), (2.8, 3.5), (3.5, 5.31), (5.31, 6.36), (6.36, 10.6), (10.62, 20)]),  # 3.5--5.3
    BlockData('钠(NA)', df['钠(NA)_mmol/L'], [(0, 108), (108, 135), (135, 146), (146, 175)]),  # 135--145
    BlockData('氯(CL)', df['氯(CL)_mmol/L'], [(0, 76.8), (76.8, 96), (96, 106), (106, 140)]),  # 96--106
    BlockData('钙(CA)', df['钙(CA)_mmol/L'], [(0, 1.8), (1.8, 2), (2, 2.55), (2.55, 3.06), (3.06, 5)]),  # 2--2.54
    BlockData('镁(MG)', df['镁(MG)_mmol/L'], [(0, 0.64), (0.64, 0.8), (0.8, 1.01), (1.01, 1.21), (1.21, 3)]),  # 0.8--1
    BlockData('无机磷测定(P)', df['无机磷测定(P)_mmol/L'], [(0, 0.56), (0.56, 0.7), (0.7, 1.451), (1.451, 1.74), (1.74, 2.91), (2.91, 6.1)]),  # 0.7--1.45
    # BlockData('(CK)磷酸肌酸激酶_U/L'),
    BlockData('(LDH)乳酸脱氢酶', df['(LDH)乳酸脱氢酶_U/L'], [(0, 87.2), (87.2, 109), (109, 246), (246, 295), (295, 490), (490, 2000)]),  # 109--245
    # BlockData(''),
    EnumData('便潜血(BOBB)', df['便潜血(BOBB)'], ['弱阳性', '阳性', '阴性']),
    # BlockData('(ALT)谷丙转氨酶_U/L'),
    BlockData('(AST)谷草转氨酶', df['(AST)谷草转氨酶_U/L'], [(0, 40.1), (40.1, 48), (48, 80), (80, 160), (160, 240)]),  # 0--40
    BlockData('(ALP)碱性磷酸酶', df['(ALP)碱性磷酸酶_U/L'], [(0, 36), (36, 45), (45, 136), (136, 162), (162, 270), (270, 540)]),  # 45--135
    BlockData('(TBA)血清总胆汁酸', df['(TBA)血清总胆汁酸_μmol/L'], [(0, 12.1), (12.1, 14.4), (14.4, 24), (24, 48), (48, 72)]),  # 0--12
    BlockData('(TP)总蛋白', df['(TP)总蛋白_g/L'], [(0, 48), (48, 60), (60, 85.1), (85.1, 102.1), (102.1, 170.1)]),  # 60--85
    BlockData('(ALB)白蛋白', df['(ALB)白蛋白_g/L'], [(0, 28), (28, 35), (35, 55.1), (55.1, 66.1), (66.1, 110)]),  # 35--55
    BlockData('(GA)糖化血清白蛋白', df['(GA)糖化血清白蛋白_%'], [(0, 8.8), (8.8, 11), (11, 16.1), (16.1, 19.3), (19.3, 32.1), (32.1, 64)]),  # 11--16
    # BlockData('(TBIL)总胆红素_μmol/L', df['(TBIL)总胆红素_μmol/L'], [(0, 2.7), (2.7, 3.4), (3.4, 20.1), (20.1, 24), (24, 40.1), (40.1, 80)]),  # 3.4--20
    # BlockData('(DBIL)直接胆红素_μmol/L', df['(DBIL)直接胆红素_μmol/L'], [(0, 1.3), (1.3, 1.7), (1.7, 8.56), (8.56, 10.3), (10.3, 17.1), (17.1, 34.3), (34.3, 51.3)]),  # 1.7--8.55
    # BlockData('(RGT)γ谷氨酰转肽酶_U/L', df['(RGT)γ谷氨酰转肽酶_U/L'], [()]) 肝炎检测指标
    # BlockData('(CHE)胆碱酯酶_kU/L', df['(CHE)胆碱酯酶_kU/L'], [()]) 肝炎
    # BlockData('(PALB)前白蛋白_g/L', df['(PALB)前白蛋白_g/L']),
    # BlockData('(PT)凝血酶原时间_S'),
    # BlockData('(APTT)活化部分凝血活酶时间_S'),
    BlockData('(DDIMER)D-二聚体定量', df['(DDIMER)D-二聚体定量_μg/l'], [(0, 201), (201, 240), (240, 401), (401, 801)]), # 0--200
    BlockData('(GLU)葡萄糖', df['(GLU)葡萄糖_mmol/L'], [(0, 3.12), (3.12, 3.9), (3.9, 6.11), (6.11, 7.32), (7.32, 12.2), (12.2, 24.4), (24.4, 36.6)]),  # 3.9--6.1
    # BlockData('(BNP)B型钠脲肽_fmol/ml', df['(BNP)B型钠脲肽_fmol/ml'], [()]) # 因地区而异
    BlockData('(PCO2)二氧化碳分压', df['(PCO2)二氧化碳分压_mmHg'], [(0, 28), (28, 35), (35, 45), (45, 54), (54, 110)]),  # 35-45
    # BlockData('(PO2)氧分压_mmHg', df['(PO2)氧分压_mmHg'])
    BlockData('(SO2)氧饱和度', df['(SO2)氧饱和度_%'], [(0, 73.5), (73.5, 91.9), (91.9, 99.1), (99.1, 110)]),  # 91.9--99
    # BlockData('(LAC)乳酸_mmol/L', df['(LAC)乳酸_mmol/L'], [()])
    # BlockData('(BEECF)细胞外剩余碱_mmol/L', df['(BEECF)细胞外剩余碱_mmol/L'], ),
    # BlockData('(BEB)剩余碱_mmol/L'),
    # BlockData('(SBC)标准碳酸氢根浓度_mmol/L'),
    # BlockData('(NCA)在PH=7.4时钙离子浓度_mmol/L'),
    # BlockData('碳酸氢根浓度_mmol/L'),
    # BlockData('(TNI)肌钙蛋白_ng/ml', df['(TNI)肌钙蛋白_ng/ml'], [()]) 没找到
    BlockData('(BASBFB)嗜碱细胞百分比', df['(BASBFB)嗜碱细胞百分比_%'], [(0, 1.01), (1.01, 1.2), (1.2, 2), (2, 4)]),  # 0--1
    BlockData('(FBG)纤维蛋白原定量', df['(FBG)纤维蛋白原定量_g/l'], [(0, 1.6), (1.6, 2), (2, 4.01), (4.01, 4.81), (4.81, 8), (8, 16)]),  # 2--4
    # BlockData('(T0390)总三碘甲状腺原氨酸_nmol/L', df['(T0390)总三碘甲状腺原氨酸_nmol/L'])
    # BlockData('(T0400)总甲状腺素_nmol/L', df['(T0400)总甲状腺素_nmol/L'], []),
    # BlockData('游三碘甲状腺原氨酸'),
    # BlockData('(T0430)游离甲状腺素_pmol/L'),
    BlockData('(HCY)同型半胱氨酸', df['(HCY)同型半胱氨酸_umol/l'], [(0, 3.2), (3.2, 4), (4, 10), (10, 15.4), (15.4, 20), (20, 30), (30, 60), (60, 120)]),  # 4-15.4
    BlockData('(NEBFB)中性粒细胞百分比', df['(NEBFB)中性粒细胞百分比_%'], [(0, 40), (40, 50), (50, 70.1), (70.1, 84.1), (84.1, 100)]),  # 50--70
    # BlockData('(TCO2)总CO2_mmol/L', df['(TCO2)总CO2_mmol/L'], [()]),
    # BlockData('阴离子间隙_mmol/L'),
]

resdf = pd.DataFrame()
for job in pendingjobs:
    subdf = job.encoding2onehot()
    resdf = pd.concat([resdf, subdf], axis=1)
    assert (len(resdf) == len(subdf))

resdf.to_csv('../data/encodingdata.txt', sep='\t', index=False)