import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

tqdm.pandas(disable=True)

from datetime import datetime

print("Start time: ", datetime.now())


mimic_data_dir = "../mimic-iii-clinical-database-1.4/"

# Get all ICU stays.
icu = pd.read_csv(
    mimic_data_dir + "ICUSTAYS.csv",
    usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
)
icu = icu.loc[icu.INTIME.notna()]
icu = icu.loc[icu.OUTTIME.notna()]

# Filter out pediatric patients.
pat = pd.read_csv(
    mimic_data_dir + "PATIENTS.csv", usecols=["SUBJECT_ID", "DOB", "DOD", "GENDER"]
)
icu = icu.merge(pat, on="SUBJECT_ID", how="left")
icu["INTIME"] = pd.to_datetime(icu.INTIME)
icu["DOB"] = pd.to_datetime(icu.DOB)
icu["AGE"] = icu.INTIME.map(lambda x: x.year) - icu.DOB.map(lambda x: x.year)
icu = icu.loc[icu.AGE >= 18]  # 53k icustays

# Extract chartevents for icu stays.
ch = []
for chunk in tqdm(
    pd.read_csv(
        mimic_data_dir + "CHARTEVENTS.csv",
        chunksize=100000,
        usecols=[
            "HADM_ID",
            "ICUSTAY_ID",
            "ITEMID",
            "CHARTTIME",
            "VALUE",
            "VALUENUM",
            "VALUEUOM",
            "ERROR",
        ],
    ), disable=True
):
    chunk = chunk.loc[chunk.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
    chunk = chunk.loc[chunk["ERROR"] != 1]
    chunk = chunk.loc[chunk.CHARTTIME.notna()]
    chunk.drop(columns=["ERROR"], inplace=True)
    ch.append(chunk)
del chunk
ch = pd.concat(ch)
ch = ch.loc[~(ch.VALUE.isna() & ch.VALUENUM.isna())]
ch["TABLE"] = "chart"
print("Chartevents read")

# Extract labevents for admissions.
la = pd.read_csv(
    mimic_data_dir + "LABEVENTS.csv",
    usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM"],
)
la = la.loc[la.HADM_ID.isin(icu.HADM_ID)]
la.HADM_ID = la.HADM_ID.astype(int)
la = la.loc[la.CHARTTIME.notna()]
la = la.loc[~(la.VALUE.isna() & la.VALUENUM.isna())]
la["ICUSTAY_ID"] = np.nan
la["TABLE"] = "lab"

# Extract bp events. Remove outliers. Make sure median values of CareVue and MetaVision items are close.
dbp = [8368, 8440, 8441, 8502, 8503, 8504, 8506, 8507, 8555, 220051, 220180, 224643, 225310, 227242]
sbp = [51, 442, 455, 3313, 3315, 3317, 3321, 3323, 6701, 220050, 220179, 224167, 225309, 227243]
mbp = [52, 224, 443, 456, 3312, 3314, 3316, 3320, 3322, 6702, 220052, 220181, 224322, 225312]

ch_bp = ch.loc[ch.ITEMID.isin(dbp + sbp + mbp)]
ch_bp = ch_bp.loc[(ch_bp.VALUENUM >= 0) & (ch_bp.VALUENUM <= 375)]
ch_bp.loc[ch_bp.ITEMID.isin(dbp), "NAME"] = "DBP"
ch_bp.loc[ch_bp.ITEMID.isin(sbp), "NAME"] = "SBP"
ch_bp.loc[ch_bp.ITEMID.isin(mbp), "NAME"] = "MBP"
ch_bp["VALUEUOM"] = "mmHg"
ch_bp["VALUE"] = None
events = ch_bp.copy()
del ch_bp

# Extract GCS events. Checked for outliers.
gcs_eye = [184, 220739]
gcs_motor = [454, 223901]
gcs_verbal = [723, 223900]
ch_gcs = ch.loc[ch.ITEMID.isin(gcs_eye + gcs_motor + gcs_verbal)]
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_eye), "NAME"] = "GCS_eye"
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_motor), "NAME"] = "GCS_motor"
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_verbal), "NAME"] = "GCS_verbal"
ch_gcs["VALUEUOM"] = None
ch_gcs["VALUE"] = None
events = pd.concat([events, ch_gcs])
del ch_gcs

# Extract heart_rate events. Remove outliers.
hr = [211, 220045]
ch_hr = ch.loc[ch.ITEMID.isin(hr)]
ch_hr = ch_hr.loc[(ch_hr.VALUENUM >= 0) & (ch_hr.VALUENUM <= 390)]
ch_hr["NAME"] = "HR"
ch_hr["VALUEUOM"] = "bpm"
ch_hr["VALUE"] = None
events = pd.concat([events, ch_hr])
del ch_hr

# Extract respiratory_rate events. Remove outliers. Checked unit consistency.
rr = [614, 615, 618, 619, 651, 3603, 220210, 224422, 224688, 224689, 224690, 227860, 227918]
ch_rr = ch.loc[ch.ITEMID.isin(rr)]
ch_rr = ch_rr.loc[(ch_rr.VALUENUM >= 0) & (ch_rr.VALUENUM <= 330)]
ch_rr["NAME"] = "RR"
ch_rr["VALUEUOM"] = "brpm"
ch_rr["VALUE"] = None
events = pd.concat([events, ch_rr])
del ch_rr

# Extract temperature events. Convert F to C. Remove outliers.
temp_c = [3655, 677, 676, 223762]
temp_f = [223761, 678, 679, 3654]
ch_temp_c = ch.loc[ch.ITEMID.isin(temp_c)]
ch_temp_f = ch.loc[ch.ITEMID.isin(temp_f)]
ch_temp_f.VALUENUM = (ch_temp_f.VALUENUM - 32) * 5 / 9
ch_temp = pd.concat([ch_temp_c, ch_temp_f])
del ch_temp_c
del ch_temp_f
ch_temp = ch_temp.loc[(ch_temp.VALUENUM >= 14.2) & (ch_temp.VALUENUM <= 47)]
ch_temp["NAME"] = "Temperature"
ch_temp["VALUEUOM"] = "C"
ch_temp["VALUE"] = None
events = pd.concat([events, ch_temp])
del ch_temp

# Extract weight events. Convert lb to kg. Remove outliers.
we_kg = [763, 224639, 226512, 226846]
we_lb = [226531]
ch_we_kg = ch.loc[ch.ITEMID.isin(we_kg)]
ch_we_lb = ch.loc[ch.ITEMID.isin(we_lb)]
ch_we_lb.VALUENUM = ch_we_lb.VALUENUM * 0.453592
ch_we = pd.concat([ch_we_kg, ch_we_lb])
del ch_we_kg
del ch_we_lb
ch_we = ch_we.loc[(ch_we.VALUENUM >= 0) & (ch_we.VALUENUM <= 300)]
ch_we["NAME"] = "Weight"
ch_we["VALUEUOM"] = "kg"
ch_we["VALUE"] = None
events = pd.concat([events, ch_we])
del ch_we

# Extract fio2 events. Convert % to fraction. Remove outliers.
fio2 = [3420, 223835, 3422, 189, 727, 190]
ch_fio2 = ch.loc[ch.ITEMID.isin(fio2)]
idx = ch_fio2.VALUENUM > 1.0
ch_fio2.loc[idx, "VALUENUM"] = ch_fio2.loc[idx, "VALUENUM"] / 100
ch_fio2 = ch_fio2.loc[(ch_fio2.VALUENUM >= 0.2) & (ch_fio2.VALUENUM <= 1)]
ch_fio2["NAME"] = "FiO2"
ch_fio2["VALUEUOM"] = None
ch_fio2["VALUE"] = None
events = pd.concat([events, ch_fio2])
del ch_fio2

# Extract capillary refill rate events. Convert to binary.
cr = [3348, 115, 8377, 224308, 223951]
ch_cr = ch.loc[ch.ITEMID.isin(cr)]
ch_cr = ch_cr.loc[~(ch_cr.VALUE == "Other/Remarks")]
idx = (ch_cr.VALUE == "Normal <3 Seconds") | (ch_cr.VALUE == "Normal <3 secs")
ch_cr.loc[idx, "VALUENUM"] = 0
idx = (ch_cr.VALUE == "Abnormal >3 Seconds") | (ch_cr.VALUE == "Abnormal >3 secs")
ch_cr.loc[idx, "VALUENUM"] = 1
ch_cr["VALUEUOM"] = None
ch_cr["NAME"] = "CRR"
events = pd.concat([events, ch_cr])
del ch_cr

# Extract glucose events. Remove outliers.
gl_bl = [225664, 1529, 811, 807, 3745, 50809]
gl_wb = [226537]
gl_se = [220621, 50931]

ev_blgl = pd.concat((ch.loc[ch.ITEMID.isin(gl_bl)], la.loc[la.ITEMID.isin(gl_bl)]))
ev_blgl = ev_blgl.loc[(ev_blgl.VALUENUM >= 0) & (ev_blgl.VALUENUM <= 2200)]
ev_blgl["NAME"] = "Glucose (Blood)"
ev_wbgl = pd.concat((ch.loc[ch.ITEMID.isin(gl_wb)], la.loc[la.ITEMID.isin(gl_wb)]))
ev_wbgl = ev_wbgl.loc[(ev_wbgl.VALUENUM >= 0) & (ev_wbgl.VALUENUM <= 2200)]
ev_wbgl["NAME"] = "Glucose (Whole Blood)"
ev_segl = pd.concat((ch.loc[ch.ITEMID.isin(gl_se)], la.loc[la.ITEMID.isin(gl_se)]))
ev_segl = ev_segl.loc[(ev_segl.VALUENUM >= 0) & (ev_segl.VALUENUM <= 2200)]
ev_segl["NAME"] = "Glucose (Serum)"

ev_gl = pd.concat((ev_blgl, ev_wbgl, ev_segl))
del ev_blgl, ev_wbgl, ev_segl
ev_gl["VALUEUOM"] = "mg/dL"
ev_gl["VALUE"] = None
events = pd.concat([events, ev_gl])
del ev_gl

# Extract bilirubin events. Remove outliers.
br_to = [50885]
br_di = [50883]
br_in = [50884]
ev_br = pd.concat(
    (
        ch.loc[ch.ITEMID.isin(br_to + br_di + br_in)],
        la.loc[la.ITEMID.isin(br_to + br_di + br_in)],
    )
)
ev_br = ev_br.loc[(ev_br.VALUENUM >= 0) & (ev_br.VALUENUM <= 66)]
ev_br.loc[ev_br.ITEMID.isin(br_to), "NAME"] = "Bilirubin (Total)"
ev_br.loc[ev_br.ITEMID.isin(br_di), "NAME"] = "Bilirubin (Direct)"
ev_br.loc[ev_br.ITEMID.isin(br_in), "NAME"] = "Bilirubin (Indirect)"
ev_br["VALUEUOM"] = "mg/dL"
ev_br["VALUE"] = None
events = pd.concat([events, ev_br])
del ev_br

# Extract intubated events.
itb = [50812]
la_itb = la.loc[la.ITEMID.isin(itb)]
idx = la_itb.VALUE == "INTUBATED"
la_itb.loc[idx, "VALUENUM"] = 1
idx = la_itb.VALUE == "NOT INTUBATED"
la_itb.loc[idx, "VALUENUM"] = 0
la_itb["VALUEUOM"] = None
la_itb["NAME"] = "Intubated"
events = pd.concat([events, la_itb])
del la_itb

# Extract multiple events. Remove outliers.
o2sat = [834, 50817, 8498, 220227, 646, 220277]
sod = [50983, 50824]
pot = [50971, 50822]
mg = [50960]
po4 = [50970]
ca_total = [50893]
ca_free = [50808]
wbc = [51301, 51300]
hct = [50810, 51221]
hgb = [51222, 50811]
cl = [50902, 50806]
bic = [50882, 50803]
alt = [50861]
alp = [50863]
ast = [50878]
alb = [50862]
lac = [50813]
ld = [50954]
usg = [51498]
ph_ur = [51491, 51094, 220734, 1495, 1880, 1352, 6754, 7262]
ph_bl = [50820]
po2 = [50821]
pco2 = [50818]
tco2 = [50804]
be = [50802]
monos = [51254]
baso = [51146]
eos = [51200]
neuts = [51256]
lym_per = [51244, 51245]
lym_abs = [51133]
pt = [51274]
ptt = [51275]
inr = [51237]
agap = [50868]
bun = [51006]
cr_bl = [50912]
cr_ur = [51082]
mch = [51248]
mchc = [51249]
mcv = [51250]
rdw = [51277]
plt = [51265]
rbc = [51279]

features = {
    "O2 Saturation": [o2sat, [0, 100], "%"],
    "Sodium": [sod, [0, 250], "mEq/L"],
    "Potassium": [pot, [0, 15], "mEq/L"],
    "Magnesium": [mg, [0, 22], "mg/dL"],
    "Phosphate": [po4, [0, 22], "mg/dL"],
    "Calcium Total": [ca_total, [0, 40], "mg/dL"],
    "Calcium Free": [ca_free, [0, 10], "mmol/L"],
    "WBC": [wbc, [0, 1100], "K/uL"],
    "Hct": [hct, [0, 100], "%"],
    "Hgb": [hgb, [0, 30], "g/dL"],
    "Chloride": [cl, [0, 200], "mEq/L"],
    "Bicarbonate": [bic, [0, 66], "mEq/L"],
    "ALT": [alt, [0, 11000], "IU/L"],
    "ALP": [alp, [0, 4000], "IU/L"],
    "AST": [ast, [0, 22000], "IU/L"],
    "Albumin": [alb, [0, 10], "g/dL"],
    "Lactate": [lac, [0, 33], "mmol/L"],
    "LDH": [ld, [0, 35000], "IU/L"],
    "SG Urine": [usg, [0, 2], ""],
    "pH Urine": [ph_ur, [0, 14], ""],
    "pH Blood": [ph_bl, [0, 14], ""],
    "PO2": [po2, [0, 770], "mmHg"],
    "PCO2": [pco2, [0, 220], "mmHg"],
    "Total CO2": [tco2, [0, 65], "mEq/L"],
    "Base Excess": [be, [-31, 28], "mEq/L"],
    "Monocytes": [monos, [0, 100], "%"],
    "Basophils": [baso, [0, 100], "%"],
    "Eosinophils": [eos, [0, 100], "%"],
    "Neutrophils": [neuts, [0, 100], "%"],
    "Lymphocytes": [lym_per, [0, 100], "%"],
    "Lymphocytes (Absolute)": [lym_abs, [0, 25000], "#/uL"],
    "PT": [pt, [0, 150], "sec"],
    "PTT": [ptt, [0, 150], "sec"],
    "INR": [inr, [0, 150], ""],
    "Anion Gap": [agap, [0, 55], "mg/dL"],
    "BUN": [bun, [0, 275], "mEq/L"],
    "Creatinine Blood": [cr_bl, [0, 66], "mg/dL"],
    "Creatinine Urine": [cr_ur, [0, 650], "mg/dL"],
    "MCH": [mch, [0, 50], "pg"],
    "MCHC": [mchc, [0, 50], "%"],
    "MCV": [mcv, [0, 150], "fL"],
    "RDW": [rdw, [0, 37], "%"],
    "Platelet Count": [plt, [0, 2200], "K/uL"],
    "RBC": [rbc, [0, 14], "m/uL"],
}

for k, v in features.items():
    print("k in features.items()", k)
    ev_k = pd.concat((ch.loc[ch.ITEMID.isin(v[0])], la.loc[la.ITEMID.isin(v[0])]))
    ev_k = ev_k.loc[(ev_k.VALUENUM >= v[1][0]) & (ev_k.VALUENUM <= v[1][1])]
    ev_k["NAME"] = k
    ev_k["VALUEUOM"] = v[2]
    ev_k["VALUE"] = None
    assert ev_k.VALUENUM.isna().sum() == 0
    events = pd.concat([events, ev_k])
del ev_k

# Free some memory.
del ch, la

# Extract outputevents.
oe = pd.read_csv(
    mimic_data_dir + "OUTPUTEVENTS.csv",
    usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUEUOM"],
)
oe = oe.loc[oe.VALUE.notna()]
oe["VALUENUM"] = oe.VALUE
oe.VALUE = None
oe = oe.loc[oe.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
oe.ICUSTAY_ID = oe.ICUSTAY_ID.astype(int)
oe["TABLE"] = "output"

# Extract information about output items from D_ITEMS.csv.
items = pd.read_csv(
    mimic_data_dir + "D_ITEMS.csv",
    usecols=["ITEMID", "LABEL", "ABBREVIATION", "UNITNAME", "PARAM_TYPE"],
)
items.loc[items.LABEL.isna(), "LABEL"] = ""
items.LABEL = items.LABEL.str.lower()
oeitems = oe[["ITEMID"]].drop_duplicates()
oeitems = oeitems.merge(items, on="ITEMID", how="left")

# Extract multiple events. Replace outliers with median.
uf = [40286]
keys = ["urine", "foley", "void", "nephrostomy", "condom", "drainage bag"]
cond = pd.concat([oeitems.LABEL.str.contains(k) for k in keys], axis=1).any(
    axis="columns"
)
ur = list(oeitems.loc[cond].ITEMID)
keys = ["stool", "fecal", "colostomy", "ileostomy", "rectal"]
cond = pd.concat([oeitems.LABEL.str.contains(k) for k in keys], axis=1).any(
    axis="columns"
)
st = list(oeitems.loc[cond].ITEMID)
ct = list(oeitems.loc[oeitems.LABEL.str.contains("chest tube")].ITEMID) + [
    226593,
    226590,
    226591,
    226595,
    226592,
]
gs = [40059, 40052, 226576, 226575, 226573, 40051, 226630]
ebl = [40064, 226626, 40491, 226629]
em = [40067, 226571, 40490, 41015, 40427]
jp = list(oeitems.loc[oeitems.LABEL.str.contains("jackson")].ITEMID)
res = [227510, 227511, 42837, 43892, 44909, 44959]
pre = [40060, 226633]

features = {
    "Ultrafiltrate": [uf, [0, 7000], "mL"],
    "Urine": [ur, [0, 2500], "mL"],
    "Stool": [st, [0, 4000], "mL"],
    "Chest Tube": [ct, [0, 2500], "mL"],
    "Gastric": [gs, [0, 4000], "mL"],
    "EBL": [ebl, [0, 10000], "mL"],
    "Emesis": [em, [0, 2000], "mL"],
    "Jackson-Pratt": [jp, [0, 2000], "ml"],
    "Residual": [res, [0, 1050], "mL"],
    "Pre-admission Output": [pre, [0, 13000], "ml"],
}

for k, v in features.items():
    print("check: loc 401, ", k)
    ev_k = oe.loc[oe.ITEMID.isin(v[0])]
    ind = (ev_k.VALUENUM >= v[1][0]) & (ev_k.VALUENUM <= v[1][1])
    med = ev_k.VALUENUM.loc[ind].median()
    ev_k.loc[~ind, "VALUENUM"] = med
    ev_k["NAME"] = k
    ev_k["VALUEUOM"] = v[2]
    events = pd.concat([events, ev_k])
del ev_k

# Extract CV and MV inputevents.
ie_cv = pd.read_csv(
    mimic_data_dir + "INPUTEVENTS_CV.csv",
    usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "AMOUNT", "AMOUNTUOM"],
)
ie_cv["TABLE"] = "input_cv"
ie_cv = ie_cv.loc[ie_cv.AMOUNT.notna()]
ie_cv = ie_cv.loc[ie_cv.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
ie_cv.CHARTTIME = pd.to_datetime(ie_cv.CHARTTIME)

ie_mv = pd.read_csv(
    mimic_data_dir + "INPUTEVENTS_MV.csv",
    usecols=["ICUSTAY_ID", "ITEMID", "STARTTIME", "ENDTIME", "AMOUNT", "AMOUNTUOM"],
)
ie_mv = ie_mv.loc[ie_mv.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]

# Split MV intervals hourly.
ie_mv.STARTTIME = pd.to_datetime(ie_mv.STARTTIME)
ie_mv.ENDTIME = pd.to_datetime(ie_mv.ENDTIME)
ie_mv["TD"] = ie_mv.ENDTIME - ie_mv.STARTTIME
new_ie_mv = ie_mv.loc[ie_mv.TD <= pd.Timedelta(1, "h")].drop(
    columns=["STARTTIME", "TD"]
)
ie_mv = ie_mv.loc[ie_mv.TD > pd.Timedelta(1, "h")]
new_rows = []
for _, row in tqdm(ie_mv.iterrows(), disable=True):
    icuid, iid, amo, uom, stm, td = (
        row.ICUSTAY_ID,
        row.ITEMID,
        row.AMOUNT,
        row.AMOUNTUOM,
        row.STARTTIME,
        row.TD,
    )
    td = td.total_seconds() / 60
    num_hours = td // 60
    hour_amount = 60 * amo / td
    for i in range(1, int(num_hours) + 1):
        new_rows.append([icuid, iid, stm + pd.Timedelta(i, "h"), hour_amount, uom])
    rem_mins = td % 60
    if rem_mins > 0:
        new_rows.append([icuid, iid, row["ENDTIME"], rem_mins * amo / td, uom])
new_rows = pd.DataFrame(
    new_rows, columns=["ICUSTAY_ID", "ITEMID", "ENDTIME", "AMOUNT", "AMOUNTUOM"]
)
new_ie_mv = pd.concat((new_ie_mv, new_rows))
ie_mv = new_ie_mv.copy()
del new_ie_mv
ie_mv["TABLE"] = "input_mv"
ie_mv.rename(columns={"ENDTIME": "CHARTTIME"}, inplace=True)
print('MV intervals done')

# Combine CV and MV inputevents.
ie = pd.concat((ie_cv, ie_mv))
del ie_cv, ie_mv
ie.rename(columns={"AMOUNT": "VALUENUM", "AMOUNTUOM": "VALUEUOM"}, inplace=True)
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)

# Convert mcg->mg, L->ml.
ind = ie.VALUEUOM == "mcg"
ie.loc[ind, "VALUENUM"] = ie.loc[ind, "VALUENUM"] * 0.001
ie.loc[ind, "VALUEUOM"] = "mg"
ind = ie.VALUEUOM == "L"
ie.loc[ind, "VALUENUM"] = ie.loc[ind, "VALUENUM"] * 1000
ie.loc[ind, "VALUEUOM"] = "ml"

# Extract Vasopressin events. Remove outliers.
vaso = [30051, 222315]
ev_vaso = ie.loc[ie.ITEMID.isin(vaso)]
ind1 = ev_vaso.VALUENUM == 0
ind2 = ev_vaso.VALUEUOM.isin(["U", "units"])
ind3 = (ev_vaso.VALUENUM >= 0) & (ev_vaso.VALUENUM <= 400)
ind = (ind2 & ind3) | ind1
med = ev_vaso.VALUENUM.loc[ind].median()
ev_vaso.loc[~ind, "VALUENUM"] = med
ev_vaso["VALUEUOM"] = "units"
ev_vaso["NAME"] = "Vasopressin"
events = pd.concat([events, ev_vaso])
del ev_vaso

# Extract Vancomycin events. Convert dose,g to mg. Remove outliers.
vanc = [225798]
ev_vanc = ie.loc[ie.ITEMID.isin(vanc)]
ind = ev_vanc.VALUEUOM.isin(["mg"])
ev_vanc.loc[ind, "VALUENUM"] = ev_vanc.loc[ind, "VALUENUM"] * 0.001
ev_vanc["VALUEUOM"] = "g"
ind = (ev_vanc.VALUENUM >= 0) & (ev_vanc.VALUENUM <= 8)
med = ev_vanc.VALUENUM.loc[ind].median()
ev_vanc.loc[~ind, "VALUENUM"] = med
ev_vanc["NAME"] = "Vancomycin"
events = pd.concat([events, ev_vanc])
del ev_vanc

# Extract Calcium Gluconate events. Convert units. Remove outliers.
cagl = [30023, 221456, 227525, 42504, 43070, 45699, 46591, 44346, 46291]
ev_cagl = ie.loc[ie.ITEMID.isin(cagl)]
ind = ev_cagl.VALUEUOM.isin(["mg"])
ev_cagl.loc[ind, "VALUENUM"] = ev_cagl.loc[ind, "VALUENUM"] * 0.001
ind1 = ev_cagl.VALUENUM == 0
ind2 = ev_cagl.VALUEUOM.isin(["mg", "gm", "grams"])
ind3 = (ev_cagl.VALUENUM >= 0) & (ev_cagl.VALUENUM <= 200)
ind = (ind2 & ind3) | ind1
med = ev_cagl.VALUENUM.loc[ind].median()
ev_cagl.loc[~ind, "VALUENUM"] = med
ev_cagl["VALUEUOM"] = "g"
ev_cagl["NAME"] = "Calcium Gluconate"
events = pd.concat([events, ev_cagl])
del ev_cagl

# Extract Furosemide events. Remove outliers.
furo = [30123, 221794, 228340]
ev_furo = ie.loc[ie.ITEMID.isin(furo)]
ind1 = ev_furo.VALUENUM == 0
ind2 = ev_furo.VALUEUOM == "mg"
ind3 = (ev_furo.VALUENUM >= 0) & (ev_furo.VALUENUM <= 250)
ind = ind1 | (ind2 & ind3)
med = ev_furo.VALUENUM.loc[ind].median()
ev_furo.loc[~ind, "VALUENUM"] = med
ev_furo["VALUEUOM"] = "mg"
ev_furo["NAME"] = "Furosemide"
events = pd.concat([events, ev_furo])
del ev_furo

# Extract Famotidine events. Remove outliers.
famo = [225907]
ev_famo = ie.loc[ie.ITEMID.isin(famo)]
ind1 = ev_famo.VALUENUM == 0
ind2 = ev_famo.VALUEUOM == "dose"
ind3 = (ev_famo.VALUENUM >= 0) & (ev_famo.VALUENUM <= 1)
ind = ind1 | (ind2 & ind3)
med = ev_famo.VALUENUM.loc[ind].median()
ev_famo.loc[~ind, "VALUENUM"] = med
ev_famo["VALUEUOM"] = "dose"
ev_famo["NAME"] = "Famotidine"
events = pd.concat([events, ev_famo])
del ev_famo

# Extract Piperacillin events. Convert units. Remove outliers.
pipe = [225893, 225892]
ev_pipe = ie.loc[ie.ITEMID.isin(pipe)]
ind1 = ev_pipe.VALUENUM == 0
ind2 = ev_pipe.VALUEUOM == "dose"
ind3 = (ev_pipe.VALUENUM >= 0) & (ev_pipe.VALUENUM <= 1)
ind = ind1 | (ind2 & ind3)
med = ev_pipe.VALUENUM.loc[ind].median()
ev_pipe.loc[~ind, "VALUENUM"] = med
ev_pipe["VALUEUOM"] = "dose"
ev_pipe["NAME"] = "Piperacillin"
events = pd.concat([events, ev_pipe])
del ev_pipe

# Extract Cefazolin events. Convert units. Remove outliers.
cefa = [225850]
ev_cefa = ie.loc[ie.ITEMID.isin(cefa)]
ind1 = ev_cefa.VALUENUM == 0
ind2 = ev_cefa.VALUEUOM == "dose"
ind3 = (ev_cefa.VALUENUM >= 0) & (ev_cefa.VALUENUM <= 2)
ind = ind1 | (ind2 & ind3)
med = ev_cefa.VALUENUM.loc[ind].median()
ev_cefa.loc[~ind, "VALUENUM"] = med
ev_cefa["VALUEUOM"] = "dose"
ev_cefa["NAME"] = "Cefazolin"
events = pd.concat([events, ev_cefa])
del ev_cefa

# Extract Fiber events. Remove outliers.
fibe = [30073, 30088, 30166, 42027, 42050, 42090, 42091, 42106, 42116, 42641, 42663, 42831, 43088, 
        43134, 43994, 44010, 44045, 44061, 44106, 44202, 44218, 44318, 44425, 44631, 44675, 44699, 44765, 
        44887, 45370, 45381, 45406, 45497, 45515, 45541, 45597, 45657, 45691, 45775, 45777, 45865, 46789, 
        225928, 225936, 226027, 226048, 226049, 226050, 226051, 227695, 227696, 227698, 227699]
ev_fibe = ie.loc[ie.ITEMID.isin(fibe)]
ind1 = ev_fibe.VALUENUM == 0
ind2 = ev_fibe.VALUEUOM == "ml"
ind3 = (ev_fibe.VALUENUM >= 0) & (ev_fibe.VALUENUM <= 1600)
ind = ind1 | (ind2 & ind3)
med = ev_fibe.VALUENUM.loc[ind].median()
ev_fibe.loc[~ind, "VALUENUM"] = med
ev_fibe["NAME"] = "Fiber"
ev_fibe["VALUEUOM"] = "ml"
events = pd.concat([events, ev_fibe])
del ev_fibe

# Extract Pantoprazole events. Remove outliers.
pant = [225910, 40549, 41101, 41583, 44008, 40700, 40550]
ev_pant = ie.loc[ie.ITEMID.isin(pant)]
ind = ev_pant.VALUENUM > 0
ev_pant.loc[ind, "VALUENUM"] = 1
ind = ev_pant.VALUENUM >= 0
med = ev_pant.VALUENUM.loc[ind].median()
ev_pant.loc[~ind, "VALUENUM"] = med
ev_pant["NAME"] = "Pantoprazole"
ev_pant["VALUEUOM"] = "dose"
events = pd.concat([events, ev_pant])
del ev_pant

# Extract Magnesium Sulphate events. Remove outliers.
masu = [222011, 30027, 227524]
ev_masu = ie.loc[ie.ITEMID.isin(masu)]
ind = ev_masu.VALUEUOM == "mg"
ev_masu.loc[ind, "VALUENUM"] = ev_masu.loc[ind, "VALUENUM"] * 0.001
ind1 = ev_masu.VALUENUM == 0
ind2 = ev_masu.VALUEUOM.isin(["gm", "grams", "mg"])
ind3 = (ev_masu.VALUENUM >= 0) & (ev_masu.VALUENUM <= 125)
ind = ind1 | (ind2 & ind3)
med = ev_masu.VALUENUM.loc[ind].median()
ev_masu.loc[~ind, "VALUENUM"] = med
ev_masu["VALUEUOM"] = "g"
ev_masu["NAME"] = "Magnesium Sulphate"
events = pd.concat([events, ev_masu])
del ev_masu

# Extract Potassium Chloride events. Remove outliers.
poch = [30026, 225166, 227536]
ev_poch = ie.loc[ie.ITEMID.isin(poch)]
ind1 = ev_poch.VALUENUM == 0
ind2 = ev_poch.VALUEUOM.isin(["mEq", "mEq."])
ind3 = (ev_poch.VALUENUM >= 0) & (ev_poch.VALUENUM <= 501)
ind = ind1 | (ind2 & ind3)
med = ev_poch.VALUENUM.loc[ind].median()
ev_poch.loc[~ind, "VALUENUM"] = med
ev_poch["VALUEUOM"] = "mEq"
ev_poch["NAME"] = "KCl"
events = pd.concat([events, ev_poch])
del ev_poch

# Extract multiple events. Remove outliers.
mida = [30124, 221668]
prop = [30131, 222168]
albu25 = [220862, 30009]
albu5 = [220864, 30008]
ffpl = [30005, 220970]
lora = [30141, 221385]
mosu = [30126, 225154]
game = [30144, 225799]
lari = [30021, 225828]
milr = [30125, 221986]
crys = [30101, 226364, 30108, 226375]
hepa = [30025, 225975, 225152]
prbc = [30001, 225168, 30104, 226368, 227070]
poin = [30056, 226452, 30109, 226377]
neos = [30128, 221749, 30127]
pigg = [226089, 30063]
nigl = [30121, 222056, 30049]
nipr = [30050, 222051]
meto = [225974]
nore = [30120, 221906, 30047]
dobu = [221653, 30306, 30042]
coll = [30102, 226365, 30107, 226376]
hyzi = [221828]
gtfl = [226453, 30059]
hymo = [30163, 221833]
fent = [225942, 30118, 221744, 30149]
inre = [30045, 223258, 30100]
inhu = [223262]
ingl = [223260]
innp = [223259]
# nana = [30140] # unknown
d5wa = [30013, 220949]
doth = [30014, 30015, 30016, 30017, 30060, 30061, 30159, 30160, 41550, 45360, 
        220950, 220952, 225823, 225825, 225827, 225941, 228140, 228141, 228142]
nosa = [225158, 30018]
hans = [30020, 225159]
stwa = [225944, 30065]
frwa = [30058, 225797, 41430, 40872, 41915, 43936, 41619, 42429, 44492, 46169, 42554]
solu = [225943]
dopa = [30043, 221662]
epin = [30119, 221289, 30044]
amio = [30112, 221347, 228339, 45402]
tpnu = [30032, 225916, 225917, 30096]
msbo = [227523]
pcbo = [227522]
prad = [30054, 226361]

features = {
    "Midazolam": [mida, [0, 500], "mg"],
    "Propofol": [prop, [0, 12000], "mg"],
    "Albumin 25%": [albu25, [0, 750], "ml"],
    "Albumin 5%": [albu5, [0, 1300], "ml"],
    "Fresh Frozen Plasma": [ffpl, [0, 33000], "ml"],
    "Lorazepam": [lora, [0, 300], "mg"],
    "Morphine Sulfate": [mosu, [0, 4000], "mg"],
    "Gastric Meds": [game, [0, 7000], "ml"],
    "Lactated Ringers": [lari, [0, 17000], "ml"],
    "Milrinone": [milr, [0, 50], "ml"],
    "OR/PACU Crystalloid": [crys, [0, 22000], "ml"],
    "Packed RBC": [prbc, [0, 17250], "ml"],
    "PO intake": [poin, [0, 11000], "ml"],
    "Neosynephrine": [neos, [0, 1200], "mg"],
    "Piggyback": [pigg, [0, 1000], "ml"],
    "Nitroglycerine": [nigl, [0, 350], "mg"],
    "Nitroprusside": [nipr, [0, 430], "mg"],
    "Metoprolol": [meto, [0, 151], "mg"],
    "Norepinephrine": [nore, [0, 80], "mg"],
    "Dobutamine": [dobu, [0, 1001], "mg"],
    "Colloid": [coll, [0, 20000], "ml"],
    "Hydralazine": [hyzi, [0, 80], "mg"],
    "GT Flush": [gtfl, [0, 2100], "ml"],
    "Hydromorphone": [hymo, [0, 125], "mg"],
    "Fentanyl": [fent, [0, 20], "mg"],
    "Insulin Regular": [inre, [0, 1500], "units"],
    "Insulin Humalog": [inhu, [0, 340], "units"],
    "Insulin Glargine": [ingl, [0, 150], "units"],
    "Insulin NPH": [innp, [0, 100], "units"],
    "D5W": [d5wa, [0, 11000], "ml"],
    "Dextrose Other": [doth, [0, 4000], "ml"],
    "Normal Saline": [nosa, [0, 11000], "ml"],
    "Half Normal Saline": [hans, [0, 2000], "ml"],
    "Sterile Water": [stwa, [0, 10000], "ml"],
    "Free Water": [frwa, [0, 2500], "ml"],
    "Solution": [solu, [0, 1500], "ml"],
    "Dopamine": [dopa, [0, 1300], "mg"],
    "Epinephrine": [epin, [0, 100], "mg"],
    "Amiodarone": [amio, [0, 1200], "mg"],
    "TPN": [tpnu, [0, 1600], "ml"],
    "Magnesium Sulfate (Bolus)": [msbo, [0, 250], "ml"],
    "KCl (Bolus)": [pcbo, [0, 500], "ml"],
    "Pre-admission Intake": [prad, [0, 30000], "ml"],
}

for k, v in features.items():
    print("check: loc 731, ", k)
    ev_k = ie.loc[ie.ITEMID.isin(v[0])]
    ind = (ev_k.VALUENUM >= v[1][0]) & (ev_k.VALUENUM <= v[1][1])
    med = ev_k.VALUENUM.loc[ind].median()
    ev_k.loc[~ind, "VALUENUM"] = med
    ev_k["NAME"] = k
    ev_k["VALUEUOM"] = v[2]
    events = pd.concat([events, ev_k])
del ev_k

# Extract heparin events. (Missed earlier.)
ev_k = ie.loc[ie.ITEMID.isin(hepa)]
ind1 = ev_k.VALUEUOM.isin(["U", "units"])
ind2 = (ev_k.VALUENUM >= 0) & (ev_k.VALUENUM <= 25300)
ind = ind1 & ind2
med = ev_k.VALUENUM.loc[ind].median()
ev_k.loc[~ind, "VALUENUM"] = med
ev_k["NAME"] = "Heparin"
ev_k["VALUEUOM"] = "units"
events = pd.concat([events, ev_k])
del ev_k

# Extract weight events from MV inputevents.
ie_mv = pd.read_csv(
    mimic_data_dir + "INPUTEVENTS_MV.csv",
    usecols=["ICUSTAY_ID", "STARTTIME", "PATIENTWEIGHT"],
)
ie_mv = ie_mv.drop_duplicates()
ie_mv = ie_mv.loc[ie_mv.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
ie_mv.rename(
    columns={"STARTTIME": "CHARTTIME", "PATIENTWEIGHT": "VALUENUM"}, inplace=True
)
ie_mv = ie_mv.loc[(ie_mv.VALUENUM >= 0) & (ie_mv.VALUENUM <= 300)]
ie_mv["VALUEUOM"] = "kg"
ie_mv["NAME"] = "Weight"
events = pd.concat([events, ie_mv])
del ie_mv

# List of all the antibiotics in mimic
antibiotics = [
    225848,
    225850,
    225851,
    225853,
    225855,
    225857,
    225859,
    225860,
    225862,
    225863,
    225865,
    225866,
    225868,
    225869,
    225871,
    225873,
    225875,
    225876,
    225877,
    225879,
    225881,
    225882,
    225883,
    225884,
    225885,
    225886,
    225888,
    225889,
    225890,
    225892,
    225893,
    225895,
    225896,
    225897,
    225898,
]

# List of antibiotics that are used in cases with sepsis
sepsis_antibiotics_codes = [
    225855,
    225893,
    225851,
    225853,
    225798,
    225859,
    225879,
    225883,
    225876,
    225842,
]

# antibiotics from prescription
sepsis_antibiotics = [
    "Ceftriaxone",
    "Ciperacillin",
    "Cefepime",
    "Ceftazidime",
    "Vancomycin HCL",
    "Ciprofloxacin",
    "Levofloxacin",
    "Meropenem",
    "Imipenem",
    "Ampicillin",
    "Piperacillin-Tazobactam Na",
]
print("check: loc 836")

prescription = pd.read_csv(
    mimic_data_dir + "PRESCRIPTIONS.csv",
    usecols=["HADM_ID", "ICUSTAY_ID", "DRUG", "STARTDATE", "ENDDATE", "ROUTE"],
)
# prescription = prescription.loc[prescription.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
prescription = prescription.loc[prescription.ROUTE == "IV"]

prescription = prescription.loc[prescription.DRUG.isin(sepsis_antibiotics)]
prescription.STARTDATE = pd.to_datetime(prescription.STARTDATE)
prescription.ENDDATE = pd.to_datetime(prescription.ENDDATE)
prescription["TD"] = prescription.ENDDATE - prescription.STARTDATE

new_prescription = prescription.loc[prescription.TD <= pd.Timedelta(1, "h")].drop(
    columns=["STARTDATE", "TD"]
)
prescription = prescription.loc[prescription.TD > pd.Timedelta(1, "h")]
new_rows = []
for _, row in tqdm(prescription.iterrows(), disable=True):
    hadm, icuid, drug, route, stm, td = (
        row.HADM_ID,
        row.ICUSTAY_ID,
        row.DRUG,
        row.ROUTE,
        row.STARTDATE,
        row.TD,
    )
    td = td.total_seconds() / 60
    num_hours = td // 60
    for i in range(1, int(num_hours) + 1):
        new_rows.append([hadm, icuid, stm + pd.Timedelta(i, "h"), drug, route])
    rem_mins = td % 60
    if rem_mins > 0:
        new_rows.append([hadm, icuid, row["ENDDATE"], drug, route])
new_rows = pd.DataFrame(
    new_rows, columns=["HADM_ID", "ICUSTAY_ID", "ENDDATE", "DRUG", "ROUTE"]
)
new_prescription = pd.concat((new_prescription, new_rows))
prescription = new_prescription.copy()
del new_prescription
prescription["TABLE"] = "prescriptions"
prescription["DRUG"] = "Antibiotics"
prescription_idx = prescription.DRUG == "Antibiotics"
prescription.loc[prescription_idx, "VALUENUM"] = 1
prescription.loc[prescription_idx, "VALUE"] = 1
prescription["VALUEUOM"] = None
prescription.rename(columns={"ENDDATE": "CHARTTIME", "DRUG": "NAME"}, inplace=True)
prescription = prescription.drop(columns=["ROUTE"])


events = pd.concat([events, prescription])
print('prescriptions done')

# Extract Mechanical Ventilation events from PROCEDUREEVENTS_MV.
procedure = pd.read_csv(
    mimic_data_dir + "PROCEDUREEVENTS_MV.csv",
    usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "ICUSTAY_ID", "STARTTIME", "ENDTIME"],
)
procedure = procedure.loc[procedure.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
mechanical_ventilation = [225792]
procedure = procedure.loc[procedure.ITEMID.isin(mechanical_ventilation)]

procedure.STARTTIME = pd.to_datetime(procedure.STARTTIME)
procedure.ENDTIME = pd.to_datetime(procedure.ENDTIME)
procedure["TD"] = procedure.ENDTIME - procedure.STARTTIME

new_procedure = procedure.loc[procedure.TD <= pd.Timedelta(1, "h")].drop(
    columns=["STARTTIME", "TD"]
)
procedure = procedure.loc[procedure.TD > pd.Timedelta(1, "h")]
new_rows = []
for _, row in tqdm(procedure.iterrows(), disable=True):
    sub_id, hadm_id, itemid, icuid, stm, td = (
        row.SUBJECT_ID,
        row.HADM_ID,
        row.ITEMID,
        row.ICUSTAY_ID,
        row.STARTTIME,
        row.TD,
    )
    td = td.total_seconds() / 60
    num_hours = td // 60
    for i in range(1, int(num_hours) + 1):
        new_rows.append([sub_id, hadm_id, itemid, icuid, stm + pd.Timedelta(i, "h")])
    rem_mins = td % 60
    if rem_mins > 0:
        new_rows.append([sub_id, hadm_id, itemid, icuid, row["ENDTIME"]])
new_rows = pd.DataFrame(
    new_rows, columns=["SUBJECT_ID", "HADM_ID", "ITEMID", "ICUSTAY_ID", "ENDTIME"]
)
new_procedure = pd.concat((new_procedure, new_rows))
procedure = new_procedure.copy()
del new_procedure
procedure.rename(columns={"ENDTIME": "CHARTTIME"}, inplace=True)
procedure["NAME"] = "Mechanically ventilated"
procedure_idx = procedure.NAME == "Mechanically ventilated"
procedure.loc[procedure_idx, "VALUENUM"] = 1
procedure.loc[procedure_idx, "VALUE"] = 1
procedure["VALUEUOM"] = None
procedure["TABLE"] = "procedures"

events = pd.concat([events, procedure])
print('procedures done')

# Extract Levofloxacin events. Convert dose,g to mg. Remove outliers. # NEW
levo = [225879]
ev_levo = ie.loc[ie.ITEMID.isin(levo)]
ind = ev_levo.VALUEUOM.isin(["mg"])
ev_levo.loc[ind, "VALUENUM"] = ev_levo.loc[ind, "VALUENUM"] * 0.001
ev_levo["VALUEUOM"] = "g"
ind = (ev_levo.VALUENUM >= 0) & (ev_levo.VALUENUM <= 8)
med = ev_levo.VALUENUM.loc[ind].median()
ev_levo.loc[~ind, "VALUENUM"] = med
ev_levo["NAME"] = "Levofloxacin"
events = pd.concat([events, ev_levo])
del ev_levo


# Save data.
events.to_csv("mimic_iii_events.csv", index=False)
icu.to_csv("mimic_iii_icu.csv", index=False)

print("The column names of events are:", events.columns)


print(
    "In the next step we use:  usecols=[HADM_ID, ICUSTAY_ID, CHARTTIME, VALUENUM, TABLE, NAME]"
)

print("Time at end of 1st script: ", datetime.now())


def f(x):
    global z_err  # loss where HADM_ID is empty
    chart_time = x.CHARTTIME
    try:
        for icu_times in x.icustay_times:
            if icu_times[1] <= chart_time <= icu_times[2]:
                return icu_times[0]
    except TypeError:
        z_err += 1


# Get ts_ind.
def inv_list(x, start=0):
    d = {}
    for i in range(len(x)):
        d[x[i]] = i
    return d


# Read extracted time series data.
events = pd.read_csv(
    "mimic_iii_events.csv",
    usecols=["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"],
)

# Convert times to type datetime.
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
icu.INTIME = pd.to_datetime(icu.INTIME)
icu.OUTTIME = pd.to_datetime(icu.OUTTIME)

# Assign ICUSTAY_ID to rows without it. Remove rows that can't be assigned one.
icu["icustay_times"] = icu.apply(lambda x: [x.ICUSTAY_ID, x.INTIME, x.OUTTIME], axis=1)
adm_icu_times = icu.groupby("HADM_ID").agg({"icustay_times": list}).reset_index()
icu.drop(columns=["icustay_times"], inplace=True)
events = events.merge(adm_icu_times, on=["HADM_ID"], how="left")
idx = events.ICUSTAY_ID.isna()

z_err = 0


events.loc[idx, "ICUSTAY_ID"] = (events.loc[idx]).progress_apply(f, axis=1)
events.drop(columns=["icustay_times"], inplace=True)
events = events.loc[events.ICUSTAY_ID.notna()]
events.drop(columns=["HADM_ID"], inplace=True)
print("(loc 1013) #z_err:", z_err)

# Filter icu table.
icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]

# Get rel_charttime in minutes.
events = events.merge(icu[["ICUSTAY_ID", "INTIME"]], on="ICUSTAY_ID", how="left")
events["rel_charttime"] = events.CHARTTIME - events.INTIME
events.drop(columns=["INTIME", "CHARTTIME"], inplace=True)
events.rel_charttime = events.rel_charttime.dt.total_seconds() // 60

# Save current icu table.
icu_full = icu.copy()

# Get icustays which lasted for atleast 24 hours.
icu = icu.loc[(icu.OUTTIME - icu.INTIME) >= pd.Timedelta(24, "h")]

# Get icustays with patient alive for atleast 24 hours.
adm = pd.read_csv(mimic_data_dir + "ADMISSIONS.csv", usecols=["HADM_ID", "DEATHTIME"])
icu = icu.merge(adm, on="HADM_ID", how="left")
icu.DEATHTIME = pd.to_datetime(icu.DEATHTIME)
icu = icu.loc[
    ((icu.DEATHTIME - icu.INTIME) >= pd.Timedelta(24, "h")) | icu.DEATHTIME.isna()
]

# Get icustays with aleast one event in first 24h.
icu = icu.loc[
    icu.ICUSTAY_ID.isin(events.loc[events.rel_charttime < 24 * 60].ICUSTAY_ID)
]

# Get sup and unsup icustays.
all_icustays = np.array(icu_full.ICUSTAY_ID)
sup_icustays = np.array(icu.ICUSTAY_ID)
unsup_icustays = np.setdiff1d(all_icustays, sup_icustays)
all_icustays = np.concatenate((sup_icustays, unsup_icustays), axis=-1)


icustay_to_ind = inv_list(all_icustays)
events["ts_ind"] = events.ICUSTAY_ID.map(icustay_to_ind)
try:
    icustay_to_ind.to_csv("icustay_to_ind.csv")
except:
    pd.DataFrame.from_dict(icustay_to_ind, orient="index", columns=["Index"]).to_csv(
        "icustay_to_ind.csv"
    )


# Rename some columns.
events.rename(
    columns={"rel_charttime": "minute", "NAME": "variable", "VALUENUM": "value"},
    inplace=True,
)

# Add gender and age.
icu_full["ts_ind"] = icu_full.ICUSTAY_ID.map(icustay_to_ind)
data_age = icu_full[["ts_ind", "AGE"]]
data_age["variable"] = "Age"
data_age.rename(columns={"AGE": "value"}, inplace=True)
data_gen = icu_full[["ts_ind", "GENDER"]]
data_gen.loc[data_gen.GENDER == "M", "GENDER"] = 0
data_gen.loc[data_gen.GENDER == "F", "GENDER"] = 1
data_gen["variable"] = "Gender"
data_gen.rename(columns={"GENDER": "value"}, inplace=True)
data = pd.concat((data_age, data_gen), ignore_index=True)
data["minute"] = 0
events = pd.concat((data, events), ignore_index=True)

# Drop duplicate events.
events.drop_duplicates(inplace=True)

# Add mortality label.
adm = pd.read_csv(
    mimic_data_dir + "ADMISSIONS.csv", usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]
)
oc = icu_full[["ts_ind", "HADM_ID", "SUBJECT_ID"]].merge(adm, on="HADM_ID", how="left")
oc = oc.rename(columns={"HOSPITAL_EXPIRE_FLAG": "in_hospital_mortality"})

# Get train-valid-test split for sup task.
all_sup_subjects = icu.SUBJECT_ID.unique()
np.random.seed(0)
np.random.shuffle(all_sup_subjects)
S = len(all_sup_subjects)
bp1, bp2 = int(0.64 * S), int(0.8 * S)
train_sub = all_sup_subjects[:bp1]
valid_sub = all_sup_subjects[bp1:bp2]
test_sub = all_sup_subjects[bp2:]
icu["ts_ind"] = icu.ICUSTAY_ID.map(icustay_to_ind)
train_ind = np.array(icu.loc[icu.SUBJECT_ID.isin(train_sub)].ts_ind)
valid_ind = np.array(icu.loc[icu.SUBJECT_ID.isin(valid_sub)].ts_ind)
test_ind = np.array(icu.loc[icu.SUBJECT_ID.isin(test_sub)].ts_ind)
print("Lengths of icu stays are:")
print("train", len(train_ind))
print("valid", len(valid_ind))
print("test", len(test_ind))

# Filter columns.
events = events[["ts_ind", "minute", "variable", "value", "TABLE"]]

# Convert minute to hour.
events["hour"] = events["minute"] / 60
events.drop(columns=["minute"], inplace=True)

# Aggregate data.
events["value"] = events["value"].astype(float)
events.loc[events["TABLE"].isna(), "TABLE"] = "N/A"
events = (
    events.groupby(["ts_ind", "hour", "variable"])
    .agg({"value": "mean", "TABLE": "unique"})
    .reset_index()
)


# Second definition of x - for Table
def f(x):
    if len(x) == 0:
        print("change in f(x) definition")
        return ""
    else:
        return ",".join(x)


events["TABLE"] = events["TABLE"].apply(f)

# Save data.
joblib.dump(
    [events, oc, train_ind, valid_ind, test_ind],
    open("mimic_iii_preprocessed.pkl", "wb"),
)

# Normalize data and save.
ts = events
means_stds = ts.groupby("variable").agg({"value": ["mean", "std"]})
means_stds.to_csv("mean_stds_variables.csv")
means_stds.columns = [col[1] for col in means_stds.columns]
means_stds.loc[means_stds["std"] == 0, "std"] = 1
means_stds.to_csv("mean_stds2_variables.csv")
ts = ts.merge(means_stds.reset_index(), on="variable", how="left")
ii = ~ts.variable.isin(["Age", "Gender"])
ts.loc[ii, "value"] = (ts.loc[ii, "value"] - ts.loc[ii, "mean"]) / ts.loc[ii, "std"]
joblib.dump(
    [ts, oc, train_ind, valid_ind, test_ind], open("mimic_iii_preprocessed.pkl", "wb")
)


print("Time at end of 2nd script: ", datetime.now())


def inv_list(l, start=0):  # Create vind
    d = {}
    for i in range(len(l)):
        d[l[i]] = i + start
    return d


def f(x):
    mask = [0 for i in range(V)]
    values = [0 for i in range(V)]
    for vv in x:  # tuple of ['vind','value']
        v = int(vv[0]) - 1  # shift index of vind
        mask[v] = 1
        values[v] = vv[1]  # get value
    return values + mask  # concat


def pad(x):
    if len(x) > 880:
        print("too many x", len(x))
    return x + [0] * (fore_max_len - len(x))


# Read data.
data = ts
# Remove test patients.
test_sub = data.loc[data.ts_ind.isin(test_ind)].ts_ind.unique()
test_data = data.loc[data.ts_ind.isin(test_sub)]
data = data.loc[~data.ts_ind.isin(test_sub)]

# Get static data with mean fill and missingness indicator.
static_varis = ["Age", "Gender"]
# i CCU-CTICU', 'i CSICU', 'i CTICU', 'i Cardiac ICU', 'i MICU', 'i Med-Surg ICU', 'i Neuro ICU', 'i SICU',
# 's Admissionheight', 's Admissionweight', 's Age', 's Gender', 's Hospitaladmitoffset', 's Hospitaladmittime24',
# 's Hospitaldischargeyear', 's Patienthealthsystemstayid', 's Unitvisitnumber']
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]  # static data are 'Age' and 'Gender'
data = data.loc[~ii]  # ~ binary flip
# print('data\n',data)

static_var_to_ind = inv_list(static_varis)  # {'Age': 0, 'Gender': 1}
D = len(static_varis)  # 17 demographic variables
N = data.ts_ind.max() + 1  # 77.704 number of patients
demo = np.zeros((int(N), int(D)))  # demo matrix
for row in tqdm(static_data.itertuples(), disable=True):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value
# print('Demo after tqdm command \n',demo[:10])
# Normalize static data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0) * 1 + (stds != 0) * stds
demo = (demo - means) / stds
# print('Demo after normalisation \n',demo[:10])
# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)  # 129 for \=12 with varis all variables except for static ones
var_to_ind = inv_list(varis, start=1)
pd.DataFrame([var_to_ind]).to_csv("var_to_ind.csv")
data["vind"] = data.variable.map(var_to_ind)
data = data[["ts_ind", "vind", "hour", "value"]].sort_values(
    by=["ts_ind", "vind", "hour"]
)
# Find max_len.
fore_max_len = 880  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip, fore_values_ip, fore_varis_ip = [], [], []
fore_op, fore_op_awesome, fore_inds = [], [], []

for w in tqdm(range(25, 124, 4)):  # range(20, 124, 4), pred_window=2
    pred_data = data.loc[(data.hour >= w) & (data.hour <= w + 24)]
    pred_data = (
        pred_data.groupby(["ts_ind", "vind"]).agg({"value": "first"}).reset_index()
    )
    pred_data["vind_value"] = pred_data[["vind", "value"]].values.tolist()
    pred_data = pred_data.groupby("ts_ind").agg({"vind_value": list}).reset_index()
    pred_data["vind_value"] = pred_data["vind_value"].apply(f)

    obs_data = data.loc[(data.hour < w) & (data.hour >= w - 24)]
    obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
    obs_data = obs_data.groupby("ts_ind").head(fore_max_len)
    obs_data = (
        obs_data.groupby("ts_ind")
        .agg({"vind": list, "hour": list, "value": list})
        .reset_index()
    )
    for pred_window in range(-24, 24, 1):
        pred_data = data.loc[
            (data.hour >= w + pred_window) & (data.hour <= w + 1 + pred_window)
        ]
        pred_data = (
            pred_data.groupby(["ts_ind", "vind"]).agg({"value": "first"}).reset_index()
        )
        pred_data["vind_value" + str(pred_window)] = pred_data[
            ["vind", "value"]
        ].values.tolist()
        pred_data = (
            pred_data.groupby("ts_ind")
            .agg({"vind_value" + str(pred_window): list})
            .reset_index()
        )
        pred_data["vind_value" + str(pred_window)] = pred_data[
            "vind_value" + str(pred_window)
        ].apply(
            f
        )  # 721 entries with 2*129 vind_values
        obs_data = obs_data.merge(pred_data, on="ts_ind")

    for col in ["vind", "hour", "value"]:
        obs_data[col] = obs_data[col].apply(pad)
    fore_op_awesome.append(
        np.array(
            list(
                [
                    list(obs_data["vind_value" + str(pred_window)])
                    for pred_window in range(-24, 24, 1)
                ]
            )
        )
    )
    fore_inds.append(np.array([int(x) for x in list(obs_data.ts_ind)]))
    fore_times_ip.append(np.array(list(obs_data.hour)))
    fore_values_ip.append(np.array(list(obs_data.value)))
    fore_varis_ip.append(np.array(list(obs_data.vind)))

del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print("The shape of fore_op", fore_op.shape)

fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]
# Generate 3 sets of inputs and outputs.
train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()  # is here an effect?
valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
fore_train_ip = [ip[train_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_valid_ip = [ip[valid_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_train_op = fore_op[train_ind]
fore_valid_op = fore_op[valid_ind]
del fore_op
# 64 min


joblib.dump(fore_train_op, "mimic_fore_train_op.pkl")
joblib.dump(fore_train_ip, "mimic_fore_train_ip.pkl")
print("Shape of fore_train_op:", fore_train_op.shape)
try:
    print("Shape of fore_train_ip:", len(fore_train_ip))    
except (TypeError, IndexError):
    print("Error: 'fore_train_ip' is either empty or improperly structured.")

joblib.dump(fore_valid_op, "mimic_fore_valid_op.pkl")
joblib.dump(fore_valid_ip, "mimic_fore_valid_ip.pkl")
print("Shape of fore_valid_op:", fore_valid_op.shape)
try:
    print("Shape of fore_valid_ip:", len(fore_valid_ip))    
except (TypeError, IndexError):
    print("Error: 'fore_valid_ip' is either empty or improperly structured.")

print("End train and valid op and ip time: ", datetime.now())


ii = test_data.variable.isin(static_varis)
static_test_data = test_data.loc[ii]  # static test_data are 'Age' and 'Gender'
test_data = test_data.loc[~ii]  # ~ binary flip

static_var_to_ind = inv_list(static_varis)  # {'Age': 0, 'Gender': 1}
D = len(static_varis)  # 17 demographic variables
N = test_data.ts_ind.max() + 1  # 77.704 number of patients
demo = np.zeros((int(N), int(D)))  # demo matrix
for row in tqdm(static_test_data.itertuples()):
    demo[int(row.ts_ind), static_var_to_ind[row.variable]] = row.value
# Normalize static test_data.
means = demo.mean(axis=0, keepdims=True)  # quite sparse
stds = demo.std(axis=0, keepdims=True)
stds = (stds == 0) * 1 + (stds != 0) * stds
demo = (demo - means) / stds
# Get variable indices.
varis = sorted(list(set(test_data.variable)))
V = len(varis)  # 129 for \=12 with varis all variables except for static ones
var_to_ind = inv_list(varis, start=1)
pd.DataFrame([var_to_ind]).to_csv('var_to_ind.csv')
test_data["vind"] = test_data.variable.map(var_to_ind)
test_data = test_data[["ts_ind", "vind", "hour", "value"]].sort_values(
    by=["ts_ind", "vind", "hour"]
)
# Find max_len.
fore_max_len = 880  # hard coded max_len of vars
# Get forecast inputs and outputs.
fore_times_ip, fore_values_ip, fore_varis_ip = [], [], []
fore_op, fore_op_awesome, fore_inds = [], [], []

for w in tqdm(range(25, 124, 4)):  # range(20, 124, 4), pred_window=2
    pred_test_data = test_data.loc[(test_data.hour >= w) & (test_data.hour <= w + 24)]
    pred_test_data = (
        pred_test_data.groupby(["ts_ind", "vind"]).agg({"value": "first"}).reset_index()
    )
    pred_test_data["vind_value"] = pred_test_data[["vind", "value"]].values.tolist()
    pred_test_data = (
        pred_test_data.groupby("ts_ind").agg({"vind_value": list}).reset_index()
    )
    pred_test_data["vind_value"] = pred_test_data["vind_value"].apply(f)

    obs_test_data = test_data.loc[(test_data.hour < w) & (test_data.hour >= w - 24)]
    obs_test_data = obs_test_data.loc[obs_test_data.ts_ind.isin(pred_test_data.ts_ind)]
    obs_test_data = obs_test_data.groupby("ts_ind").head(fore_max_len)
    obs_test_data = (
        obs_test_data.groupby("ts_ind")
        .agg({"vind": list, "hour": list, "value": list})
        .reset_index()
    )
    for pred_window in range(-24, 24, 1):
        pred_test_data = test_data.loc[
            (test_data.hour >= w + pred_window)
            & (test_data.hour <= w + 1 + pred_window)
        ]
        pred_test_data = (
            pred_test_data.groupby(["ts_ind", "vind"])
            .agg({"value": "first"})
            .reset_index()
        )
        pred_test_data["vind_value" + str(pred_window)] = pred_test_data[
            ["vind", "value"]
        ].values.tolist()
        pred_test_data = (
            pred_test_data.groupby("ts_ind")
            .agg({"vind_value" + str(pred_window): list})
            .reset_index()
        )
        pred_test_data["vind_value" + str(pred_window)] = pred_test_data[
            "vind_value" + str(pred_window)
        ].apply(
            f
        )  # 721 entries with 2*129 vind_values
        obs_test_data = obs_test_data.merge(pred_test_data, on="ts_ind")

    for col in ["vind", "hour", "value"]:
        obs_test_data[col] = obs_test_data[col].apply(pad)
    fore_op_awesome.append(
        np.array(
            list(
                [
                    list(obs_test_data["vind_value" + str(pred_window)])
                    for pred_window in range(-24, 24, 1)
                ]
            )
        )
    )
    fore_inds.append(np.array([int(x) for x in list(obs_test_data.ts_ind)]))
    fore_times_ip.append(np.array(list(obs_test_data.hour)))
    fore_values_ip.append(np.array(list(obs_test_data.value)))
    fore_varis_ip.append(np.array(list(obs_test_data.vind)))

del test_data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)

fore_op_awesome = np.concatenate(fore_op_awesome, axis=1)
fore_op = np.swapaxes(fore_op_awesome, 0, 1)
print("The shape of fore_op", fore_op.shape)

fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]
# Generate 3 sets of inputs and outputs.
test_ind = np.argwhere(np.in1d(fore_inds, test_ind)).flatten()
fore_test_ip = [
    ip[test_ind] for ip in [fore_demo, fore_times_ip, fore_values_ip, fore_varis_ip]
]
del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_test_op = fore_op[test_ind]
del fore_op
# 64 min

joblib.dump(fore_test_op, "mimic_fore_full_test_op.pkl")
joblib.dump(fore_test_ip, "mimic_fore_full_test_ip.pkl")
print("Shape of fore_test_op:", fore_test_op.shape)
print("Shape of fore_test_op:", fore_valid_op.shape)
try:
    print("Shape of fore_test_ip:", len(fore_test_ip))    
except (TypeError, IndexError):
    print("Error: 'fore_train_ip' is either empty or improperly structured.")


print("V", V, "D", D, "fore_max_len", fore_max_len, "should be (131,2,880)")


print("End time: ", datetime.now())
