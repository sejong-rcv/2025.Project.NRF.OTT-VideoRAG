import json
import os
from collections import defaultdict
from collections import Counter
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import pandas as pd
from glob import glob

results_path = "{}/generate_videomme.json"
OPTIONS = ["A", "B", "C", "D"]

def read_json(path): 
    with open(path) as f:
        data = json.load(f)
    return data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def entropy(p):
    # epsilon = 1e-10
    # H = -(p * (p+epsilon).log()).sum()
    H = -(p * np.log(p)).sum()
    return H

def convert_id_to_ans(test_raw_data):
    test_id_to_answer = {}
    for vrow in test_raw_data:
        for qrow in vrow['questions']:
            test_id_to_answer[str(qrow['question_id'])] = qrow['answer']
    return test_id_to_answer

def convert_info_to_dict(target_info):
    video_id_to_info = {}
    for info_i in target_info:
        video_id_to_info[info_i['video_id']] = info_i
    return video_id_to_info

def cal_acc(test_raw_data, logits_type):
    res = {}
    task_result = defaultdict(dict)
    for vidx, vrow in enumerate(test_raw_data):
        for qidx, qrow in enumerate(vrow['questions']):
            truth_answer = qrow['answer']
            pred = qrow[logits_type][0]
            pred_answer = OPTIONS[np.argmax(pred)]

            task_succ_flag = 0
            if pred_answer == truth_answer:
                res[qrow['question_id']] = 1
                task_succ_flag = 1
            else:
                res[qrow['question_id']] = 0
            if qrow['task_type'] in task_result[vrow['duration']]:
                task_result[vrow['duration']][qrow['task_type']].append(task_succ_flag)
            else:
                task_result[vrow['duration']][qrow['task_type']] = [task_succ_flag]
    
    return np.mean([v for k, v in res.items()]), task_result, res

def cal_coverage(pred_set, test_id_to_answer):
    cover = []
    for k, v in pred_set.items():
        if test_id_to_answer[k] in v:
            cover.append(1)
        else:
            cover.append(0)
    coverage_score = sum(cover) / len(cover)
    return coverage_score
    
def cal_set_size(pred_set, video_id_to_info):
    sz = {}
    for k, v in pred_set.items():
        sz[k] = len(v)
    return sz

def cal_uacc(acc, set_size):
    uacc = acc * np.sqrt(len(OPTIONS)) / set_size
    return uacc

def LAC_CP(cal_raw_data, test_raw_data, alpha=0.1):
    pred_set_sampling = {}
    for logits_type_i in logits_type:
        # construct calibration
        cal_scores = []
        for idx, raw in enumerate(cal_raw_data):
            for qidx, qraw in enumerate(raw['questions']):
                probs = softmax(qraw[logits_type_i][0])
                truth_answer = qraw['answer']
                cal_scores.append(1-probs[OPTIONS.index(truth_answer)])
        n = len(cal_scores)
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(cal_scores, q_level, method='higher')
        # make prediction set
        pred_set = {}
        for idx, raw in enumerate(test_raw_data):
            for qidx, qraw in enumerate(raw['questions']):
                probs = softmax(qraw[logits_type_i][0])
                ps = []
                for ii, p in enumerate(probs):
                    if p >= 1-qhat:
                        ps.append(OPTIONS[ii])
                if len(ps) == 0:
                    ps.append(OPTIONS[np.argmax(probs)])
                pred_set[str(qraw['question_id'])] = ps
        pred_set_sampling[logits_type_i] = pred_set
    return pred_set_sampling

def APS_CP(cal_raw_data, test_raw_data, alpha=0.1):
    pred_set_sampling = {}
    for logits_type_i in logits_type:
        # construct calibration
        cal_scores = []
        for idx, raw in enumerate(cal_raw_data):
            for qidx, qraw in enumerate(raw['questions']):
                probs = softmax(qraw[logits_type_i][0])
                truth_answer = qraw['answer']
                cal_pi = np.argsort(probs)[::-1] # descending order
                cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                cal_sum_r = np.take_along_axis(cal_sum, cal_pi.argsort(), axis=0)
                cal_score = cal_sum_r[OPTIONS.index(truth_answer)]
                cal_scores.append(cal_score)
        
        n = len(cal_scores)
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(cal_scores, q_level, method='higher')
        # make prediction set
        pred_set = {}
        for idx, raw in enumerate(test_raw_data):
            for qidx, qraw in enumerate(raw['questions']):
                probs = softmax(qraw[logits_type_i][0])
                cal_pi = np.argsort(probs)[::-1] # descending order
                cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                ps = []
                ii = 0
                while ii < len(cal_sum) and cal_sum[ii] <= qhat:
                    op_id = cal_pi[ii]
                    ps.append(OPTIONS[op_id])
                    ii += 1
                if len(ps) == 0:
                    op_id = cal_pi[ii]
                    ps.append(OPTIONS[op_id])
                pred_set[str(qraw['question_id'])] = ps
        pred_set_sampling[logits_type_i] = pred_set
    
    return pred_set_sampling

def get_entropy(raw_data, logits_type):
    entropy_values_sampling = {}    
    for logits_type_i in logits_type:
        entropy_values = {}
        for idx, raw in enumerate(raw_data):
            for qidx, qraw in enumerate(raw['questions']):
                prob = softmax(qraw[logits_type_i][0])
                H = entropy(prob)
                entropy_values[qraw['question_id']] = H
        entropy_values_sampling[logits_type_i] = entropy_values
    
    return entropy_values_sampling

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--target_dir", help="the target path to read the results")
    # args = parser.parse_args()

    # target_path = results_path.format(args.target_dir)
    # target_dirs = glob("results1125*")
    target_dirs = glob("results_1127*")
    target_dirs = sorted(target_dirs)
    for target_dir in target_dirs:
        target_path = results_path.format(target_dir)
        if not os.path.exists(target_path):
            continue
        target_info = read_json(target_path)
        # if sum([1 for info_i in target_info if info_i['duration'] == 'long']) < 300:
        #     print(f'\n>> {target_dir} is not done.')
        #     print(f"The number: {sum([1 for info_i in target_info if info_i['duration'] == 'long']) }")
        #     continue

        cal_raw_data, test_raw_data = train_test_split(target_info, train_size=0.5, random_state=42)
        test_id_to_answer = convert_id_to_ans(test_raw_data)
        video_id_to_info = convert_info_to_dict(target_info)
        
        logits_type = ['logits_16', 'logits_20',  'logits_24', 'logits_28', 'logits']
        test_entropy = get_entropy(test_raw_data, logits_type)
        cali_entropy = get_entropy(cal_raw_data, logits_type)
        all_entropy = get_entropy(target_info, logits_type)
        pred_set_LAC = LAC_CP(cal_raw_data, test_raw_data)
        pred_set_APS = APS_CP(cal_raw_data, test_raw_data)
        
        test_entropy_results = {}
        cali_entropy_results = {}
        all_entropy_results = {}
        LAC_coverage = {}
        LAC_set_size = {}
        LAC_results = {}
        APS_coverage = {}
        APS_set_size = {}
        APS_results = {}
        for logits_type_i in logits_type:
            acc, _, acc_detail_test = cal_acc(test_raw_data, logits_type_i)
            cali_acc, _, _ = cal_acc(cal_raw_data, logits_type_i)
            acc_all, all_task_result, _ = cal_acc(target_info, logits_type_i)

            # target_info
            # CP method LAC
            test_entropy_results[logits_type_i] = np.mean([v for k, v in test_entropy[logits_type_i].items()])
            cali_entropy_results[logits_type_i] = np.mean([v for k, v in cali_entropy[logits_type_i].items()])
            all_entropy_results[logits_type_i] = np.mean([v for k, v in all_entropy[logits_type_i].items()])
            
            coverage_LAC = cal_coverage(pred_set_LAC[logits_type_i], test_id_to_answer)
            set_size_LAC = cal_set_size(pred_set_LAC[logits_type_i], video_id_to_info)
            uacc_LAC = cal_uacc(acc, np.mean([v for k, v in set_size_LAC.items()]))
            
            coverage_APS = cal_coverage(pred_set_APS[logits_type_i], test_id_to_answer)
            set_size_APS = cal_set_size(pred_set_APS[logits_type_i], video_id_to_info)
            uacc_APS = cal_uacc(acc, np.mean([v for k, v in set_size_APS.items()]))

            LAC_coverage[logits_type_i] = coverage_LAC
            LAC_set_size[logits_type_i] = np.mean([v for k, v in set_size_LAC.items()])
            LAC_results[logits_type_i] = uacc_LAC
            APS_coverage[logits_type_i] = coverage_APS
            APS_set_size[logits_type_i] = np.mean([v for k, v in set_size_APS.items()])
            APS_results[logits_type_i] = uacc_APS
        
            # 결과를 딕셔너리로 정리
            results_task = defaultdict(list)
            results_task["Metric"] = ["short", "medium", "long", "average"]
            for k, v in all_task_result['long'].items():
                result_list = []
                scores = []
                for duration in ["short", "medium", "long"]:
                    score = np.mean(all_task_result[duration][k])
                    scores.append(score)
                    total_num = len(all_task_result[duration][k])
                    # wrong_num = total_num - np.sum(all_task_result[duration][k])
                    corr_num = np.sum(all_task_result[duration][k])
                    result_text = str(f"{round(score, 3)} ({int(corr_num)}/{int(total_num)})")
                    result_list.append(result_text)
                result_list.append(round(np.mean(scores), 4))
                results_task[k] = result_list
            result_avg = [ np.concatenate([v for k, v in all_task_result[duration].items()]).mean() for duration in ["short", "medium", "long"]]
            result_avg.append(acc_all)
            results_task["Average"] = result_avg

            results_uncertainty_all = defaultdict(list)
            # results_uncertainty_all["Metric"] = ["short entropy", "medium entropy", "long entropy",      "short set size", "medium set size", "long set size",      "short acc", "medium acc", "long acc"]
            results_uncertainty_all["Metric"] = ["short", "medium", "long"]
            entropy_scores_family = defaultdict(list)
            set_size_LAC_family = defaultdict(list)
            set_size_APS_family = defaultdict(list)
            acc_family = defaultdict(list)
            for k, v in test_entropy[logits_type_i].items():
                duration = video_id_to_info[k.split('-')[0]]['duration']
                entropy_scores_family[duration].append(v) 
                set_size_LAC_family[duration].append(set_size_LAC[k])
                set_size_APS_family[duration].append(set_size_APS[k])
                acc_family[duration].append(acc_detail_test[k])

            results_uncertainty_all["Entropy"] = [ np.mean(entropy_scores_family[duration]) for duration in ["short", "medium", "long"]]
            results_uncertainty_all["Set size LAC"] = [ np.mean(set_size_LAC_family[duration]) for duration in ["short", "medium", "long"]]
            results_uncertainty_all["Set size APS"] = [ np.mean(set_size_APS_family[duration]) for duration in ["short", "medium", "long"]]
            results_uncertainty_all["Accuracy"] = [ np.mean(acc_family[duration]) for duration in ["short", "medium", "long"]]
            
            
            # 결과를 딕셔너리로 정리
            results_total = {
                "Metric": ["Calibration", "Test", "Total"],
                "Acc ↑": [cali_acc, acc, acc_all],
                "Entropy ↓": [cali_entropy_results[logits_type_i], test_entropy_results[logits_type_i], all_entropy_results[logits_type_i]]
            }

            results_uncertainty = {
                "Metric": ["Coverage ↑", "Set Size ↓", "UACC ↑"],
                "LAC": [coverage_LAC, LAC_set_size[logits_type_i], uacc_LAC],
                "APS": [coverage_APS, APS_set_size[logits_type_i], uacc_APS]
            }

            # pandas DataFrame 생성
            df_total = pd.DataFrame(results_total)
            df_cali_uncert = pd.DataFrame(results_uncertainty)
            df_task = pd.DataFrame(results_task)
            df_task_uncertainty = pd.DataFrame(results_uncertainty_all)
            pd.options.display.float_format = "{:.5f}".format

            # 테이블 출력
            print(f'\n>> {target_dir} ({logits_type_i})')
            print(df_total.to_string(index=False))
            print(df_cali_uncert.to_string(index=False))
            print(df_task.to_string(index=False))
            print(df_task_uncertainty.to_string(index=False))
            import pdb;pdb.set_trace()

        # # # 구분자 행 추가 (섹션 이름 넣기)
        # # df_total_section = df_total.copy()
        # # df_total_section.insert(0, "Section", "Overall")

        # # df_uncert_section = df_cali_uncert.copy()
        # # df_uncert_section.insert(0, "Section", "Uncertainty")

        # # df_task_section = df_task.copy()
        # # df_task_section.insert(0, "Section", "Task")

        # # # 하나의 DataFrame으로 연결
        # # df_all = pd.concat([
        # #     df_total_section,
        # #     df_uncert_section,
        # #     df_task_section
        # # ], axis=0)

        # combined_rows = []

        # # Overall
        # combined_rows.append({"Section": f"===== {target_dir} : Overall ====="})
        # combined_rows.append({})
        # combined_rows.extend(df_total.to_dict('records'))
        # combined_rows.append({})
        # combined_rows.append({})

        # # Uncertainty
        # combined_rows.append({"Section": f"===== {target_dir} : Uncertainty ====="})
        # combined_rows.append({})
        # combined_rows.extend(df_cali_uncert.to_dict('records'))
        # combined_rows.append({})
        # combined_rows.append({})

        # # Task
        # combined_rows.append({"Section": f"===== {target_dir} : Task Results ====="})
        # combined_rows.append({})
        # combined_rows.extend(df_task.to_dict('records'))
        # combined_rows.append({})
        # combined_rows.append({})

        # df_all = pd.DataFrame(combined_rows)

        # # CSV 저장
        # output_dir = "results_csv"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # df_all.to_csv(os.path.join(output_dir, f"{target_dir}.csv"), index=False)

        
