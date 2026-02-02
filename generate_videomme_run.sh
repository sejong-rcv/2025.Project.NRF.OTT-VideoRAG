# # python generate_videomme.py
# # python generate_videomme_base8.py
# # python generate_videomme_base16.py
# # python generate_videomme_base32.py
# # python generate_videomme_longvideo_v2.py --max_frames_num 32 --results_path result1117_llavanext_32fr_v2


# # =====================================
python generate_videomme_logits_frame_uncertainty.py --max_frames_num 32 --results_path results_1127_llavavideo_frame_sampling_uncertainty_rag_original --use_ocr --use_asr --use_det
python generate_videomme_longvideo_get_logits_frame_uncertainty.py --max_frames_num 32 --results_path results_1127_longvideo_frame_sampling_uncertainty_rag_original --use_ocr --use_asr --use_det