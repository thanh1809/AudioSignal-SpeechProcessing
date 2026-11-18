from pathlib import Path
import json,csv
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
cross=pd.read_csv(ROOT/'results/crossfade_experiment.csv')
lat=pd.read_csv(ROOT/'results/test_sentences.csv')
D=Document(); sec=D.sections[0]
sec.top_margin=Inches(.55);sec.bottom_margin=Inches(.55);sec.left_margin=Inches(.68);sec.right_margin=Inches(.68)
styles=D.styles;styles['Normal'].font.name='Aptos';styles['Normal'].font.size=Pt(9);styles['Normal'].font.color.rgb=RGBColor(40,48,58)
for n,size,col in [('Title',23,'15324B'),('Heading 1',15,'15324B'),('Heading 2',11,'286A8A')]:
    st=styles[n];st.font.name='Aptos Display';st.font.size=Pt(size);st.font.bold=True;st.font.color.rgb=RGBColor.from_string(col)

def p(txt='',style=None):
    q=D.add_paragraph(style=style);q.add_run(txt);q.paragraph_format.space_after=Pt(5);return q
def img(name,width=6.1,cap=None):
    q=D.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.add_run().add_picture(str(ROOT/'results/figures'/name),width=Inches(width))
    if cap:
        c=D.add_paragraph();c.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=c.add_run(cap);r.italic=True;r.font.size=Pt(8)
def shade(c,color):
    shd=OxmlElement('w:shd');shd.set(qn('w:fill'),color);c._tc.get_or_add_tcPr().append(shd)
def table(headers,rows):
    t=D.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1';t.alignment=1
    for c,h in zip(t.rows[0].cells,headers):c.text=str(h);shade(c,'15324B')
    for c in t.rows[0].cells:
        for r in c.paragraphs[0].runs:r.font.bold=True;r.font.color.rgb=RGBColor(255,255,255);r.font.size=Pt(8)
    for row in rows:
        cc=t.add_row().cells
        for c,v in zip(cc,row):c.text=str(v)
        for c in cc:
            for para in c.paragraphs:
                para.paragraph_format.space_after=Pt(0)
                for run in para.runs:run.font.size=Pt(8)
    return t
def br():D.add_page_break()

D.add_paragraph('Lab 4 Mini TTS',style='Title')
p('Báo cáo thực hành CSE457 | Tổng hợp tiếng nói')
p('Pipeline dùng inventory, unit selection, cross-fade, prosody, HMM-state parameters và neural VITS để sinh câu thông báo xếp hàng. Inventory có 16 nhãn; queue case chạy qua cả ba backend waveform. MOS chủ quan vẫn cần người nghe chấm qua phiếu ratings.csv.')
D.add_heading('1. Inventory và phạm vi dữ liệu',level=1)
p('Nguồn gốc có bảy từ lệnh: bật, dừng, lên, phải, trái, tắt, xuống. WAV trong dataset/ bị sao chép nguyên byte trong từng nhãn, nên năm candidate cho mỗi từ lấy từ dataset_aug/. Đây là biến thể augmentation, không phải năm lần thu tự nhiên độc lập. Chín từ còn thiếu trong câu queue (mời, khách, hàng, số, hai, ba, đến, quầy, năm) được tổng hợp thành hai biến thể mỗi từ bằng checkpoint Meta MMS VITS tiếng Việt. Metadata ghi nguồn recording/model cho từng file.')
p(f"Inventory cuối có {M['inventory_label_count']} nhãn và {M['candidate_recording_count']} WAV candidates. Câu mẫu đã được normalize và chạy đủ. Đầu ra cho thấy khác biệt giữa các từ thu thật và các unit được tổng hợp; các unit tổng hợp không được mô tả như bản thu người nói độc lập.")
img('inventory_stats.png',6.0,'Hình 1. Phân bố thời lượng và RMS của inventory sau trim.')
D.add_heading('Text normalization',level=2)
p('normalize_text() viết thường, bỏ dấu câu và đọc từng chữ số riêng. Ví dụ “23” → “hai ba”; toàn bộ token của câu mẫu hiện có unit WAV sau khi bổ sung chín từ còn thiếu.')
table(['ID','Câu demo','Unit được phát'],[[r.id,r.text,r.tokens] for r in lat.itertuples()])

br();D.add_heading('2. Ghép nối và unit selection',level=1)
p('Baseline chọn candidate 01 cho mỗi token, không cross-fade và lưu audio_output/raw_concat.wav. Target cost đo z-score duration và log-RMS so với thống kê của nhãn. Join cost tính sai khác log-magnitude phổ trung bình ở hai cửa sổ 25 ms và chênh log-RMS: C_join = mean((L_left-L_right)^2) + 12 × |log(RMS_left/RMS_right)|. Hệ số 12 cân bằng năng lượng với sai khác phổ.')
p('Dynamic programming tối thiểu hóa tổng target cost và join cost. Trên queue sentence, đường tối ưu trùng đường candidate đầu tiên nên join cost không đổi; không phải lúc nào DP cũng giảm giá trị so với baseline. Trên câu S1 gồm các bản thu augmented, selection giảm join cost %.1f%%. Chỉ tối ưu join có thể chọn cặp phổ khớp nhưng sai target/context.'%M['command_join_reduction_percent'])
table(['Utterance','Cách chọn','Total join cost'],[[r.utterance,r.method,f"{r.total_join_cost:.1f}"] for r in pd.read_csv(ROOT/'results/unit_selection_summary.csv').itertuples()])
img('unit_selection_costs.png',5.7,'Hình 2. Join cost giữa các cặp candidate ở từng ranh giới.')
D.add_heading('Cross-fade và boundary check',level=2)
table(['Cross-fade','Mean sample jump','Mean energy jump','Chỉ số chuẩn hóa'],[[f"{r.crossfade_ms} ms",f"{r.mean_sample_jump:.5f}",f"{r.mean_energy_jump:.3f}",f"{r.combined_discontinuity_proxy:.3f}"] for r in cross.itertuples()])
p(f"Chọn {M['crossfade_chosen_ms']} ms theo tổng tỉ số sample jump và energy jump đã chuẩn hóa so với baseline 0 ms của queue sentence. Fade dài có thể làm mờ phụ âm; lựa chọn dựa trên hai chỉ số discontinuity, không phải MOS. best_concat.wav là chuỗi DP với fade đã chọn.")
img('boundary_comparison.png',6.0,'Hình 3. Waveform và spectrogram gần cùng boundary trước/sau xử lý.')

br();D.add_heading('3. Prosody và source-filter',level=1)
p('For the voiced unit len, median F0 is {0:.1f} Hz. At Fs=16 kHz, F0=Fs/Np gives Np about {1} samples. Low-energy frames are excluded from YIN. Duration stretch trials use 0.8x, 1.0x, and 1.2x; pitch trials use -2, 0, and +2 semitones. In the source-filter view, F0 belongs to the periodic excitation while formants mainly belong to the vocal-tract spectral envelope.'.format(M['median_f0_hz'],M['pitch_period_samples']))
p('Cross-fade/OLA trộn hai waveform theo ramp; phase và lệch thời điểm có thể khiến từng biên không giảm đều cả sample jump lẫn energy jump. TD-PSOLA thay pitch bằng cách đặt lại các đoạn quanh pitch epoch với khoảng cách mới rồi overlap-add; stretching hữu thanh quá mức làm lặp chu kỳ, có thể gây buzziness, nhất là fricative hữu thanh.')
img('prosody_f0_pitch.png',5.9,'Hình 4. F0 và đáp ứng phổ tương đối ở ba mức pitch shift.')
D.add_heading('4. HMM parameter-generation demo',level=1)
p('Mỗi word được chia thành 5 state theo thời gian. State parameters CSV lưu mean duration, F0, RMS và spectral centroid từ recording/units của VITS; câu queue lặp state sequence trái-sang-phải và tạo trajectory. Đây là state-parameter demonstration theo hướng dẫn Lab, không phải HMM acoustic model được tối ưu độc lập. hmm_tts.wav là hybrid: state duration điều khiển time-stretch, còn âm sắc đến từ waveform units; backend không có vocoder tham số riêng.')
img('hmm_tracks.png',5.9,'Hình 5. F0 và spectral-centroid trajectory theo state.')
p('Khác biệt chính: concatenative TTS phát lại đoạn waveform tự nhiên nên phụ thuộc inventory/context; parametric HMM sinh trajectory từ phân bố state, dễ điều khiển duration/feature nhưng có thể nghe mờ/robotic khi model nhỏ. Delta features ràng buộc biến thiên giữa frame và giúp trajectory trơn hơn.')

br();D.add_heading('5. Đánh giá ba backend',level=1)
p('Các output dùng cùng câu queue khi so waveform/spectrogram. Sample/energy discontinuity là continuity proxy; token coverage xác nhận mỗi từ có unit, nhưng không thay thế human intelligibility test. MOS/naturalness chủ quan chưa có kết quả; results/ratings.csv là phiếu trống cho listening test, không chứa điểm giả.')
table(['Backend','Intelligibility','Naturalness','Continuity','Latency / tình trạng'],[
 ['Concatenative','Unit cho đủ từng token; listener test pending','Chưa rating','Boundary metrics','Text→DP→audio: %.3f s'%lat[lat.id=='QUEUE'].generation_latency_s.iloc[0]],
 ['HMM state hybrid','Cùng token source units','Chưa rating','Waveform/spectrogram','Có output; hybrid, không full vocoder'],
 ['Neural MMS VITS','Sinh trực tiếp từ cả câu','Chưa rating','Waveform/mel có lưu','Inference %.3f s'%M['neural_latency_s']['QUEUE']]])
p('Đã chạy checkpoint facebook/mms-tts-vie của Meta bằng Transformers/PyTorch ở local-only mode; model trả waveform 16 kHz cho ba câu thử và câu queue. Model card: huggingface.co/facebook/mms-tts-vie; Pratap et al., 2023, arXiv:2305.13516. Weights nằm trong cache, không đóng gói vào project. License CC-BY-NC 4.0: chỉ dùng phi thương mại và giữ attribution. Latency là inference time, không gồm thời gian load model.')
img('backend_comparison.png',5.7,'Hình 6. Raw concat, unit-selection, HMM hybrid và neural VITS trên cùng câu queue.')
img('neural_queue_mel.png',5.7,'Hình 7. Mel-spectrogram từ neural VITS cho câu queue.')

br();D.add_heading('6. Trả lời câu hỏi báo cáo',level=1)
qa=[
('1. Vì sao đoạn tự nhiên vẫn nghe không tự nhiên?','Đoạn có thể khác speaker, pitch, accent, level, coarticulation và prosody. Boundary cắt sai hoặc chọn unit không đúng ngữ cảnh làm câu rời rạc.'),
('2. Target cost và join cost khác gì?','Target cost đo độ phù hợp candidate với vị trí/từ mong muốn (ở đây duration và RMS); join cost đo discontinuity giữa hai candidate. Chỉ tối ưu join có thể chọn chuỗi nối êm nhưng sai unit, hoặc lặp candidate có âm sắc lệch target.'),
('3. Vì sao cross-fade dài làm mờ phụ âm?','Nó trộn hai đoạn qua thời gian dài, làm chồng lấp burst/transition và triệt tiêu một phần tín hiệu do phase khác nhau.'),
('4. F0 và formant theo source-filter?','Nguồn excitation quyết định chu kỳ/F0; bộ lọc vocal tract quyết định envelope cộng hưởng/formant. Pitch shift lý tưởng thay nguồn, không dịch các cộng hưởng.'),
('5. PSOLA và buzziness?','PSOLA overlap-add các frame quanh pitch epoch ở chu kỳ mới. Kéo voiced fricative quá dài lặp excitation tuần hoàn trên thành phần vốn nhiễu, tạo âm buzzing.'),
('6. Delta giúp HMM trơn thế nào?','Delta mô tả đạo hàm feature; objective ML có thể ràng buộc cả static và dynamic trajectory, tránh nhảy state-level giữa frame.'),
('7. Vì sao waveform concat cần inventory lớn?','Cần nhiều context, speaker, tốc độ, biên và prosody để tìm unit hợp mục tiêu mà vẫn nối mượt.'),
('8. Concatenative và neural TTS?','Concatenative chọn/lắp waveform lưu sẵn, dễ nghe tự nhiên trong miền thu nhưng cần kho unit lớn. Neural dự đoán acoustic representation/vocoder từ text, tổng quát linh hoạt hơn nhưng cần dữ liệu/model và tài nguyên inference.'),
('9. Vì sao không tải model tùy ý?','Khác kiến trúc, tiếng, checkpoint, version và thiết bị làm so sánh latency/quality không công bằng; offline packaging bảo đảm tái lập và phù hợp yêu cầu buổi Lab.'),
('10. MOS cao nhưng intelligibility thấp?','Có thể dùng cho nhạc nền/ambient hoặc trải nghiệm không cần hiểu nội dung; không phù hợp thông báo, hướng dẫn hay cảnh báo cần truyền đúng thông tin.')]
for q,a in qa:
    z=D.add_paragraph();z.paragraph_format.space_after=Pt(4);r=z.add_run(q+' ');r.bold=True;z.add_run(a)
D.add_heading('Kết luận và giới hạn',level=2)
p('Đã chạy đủ phần kỹ thuật của queue case: 16 unit labels, 53 waveform candidates, raw/best/HMM-hybrid/neural VITS audio, spectrograms, mel, state tracks, DP và latency. Còn bước duy nhất cần người nghe là điền ratings.csv (5 người nghe đề xuất) để tính MOS, naturalness, continuity và human intelligibility; các trường này không thể suy ra trung thực từ waveform statistics.')
p('Tái lập: cài requirements.txt rồi chạy `python lab4.py` từ thư mục LAB4. Báo cáo, bảng ratings trống, model availability và artifacts đầu ra nằm trong cùng thư mục.')
D.core_properties.title='Lab 4 Mini TTS';D.core_properties.subject='CSE457 speech synthesis report';D.save(ROOT/'Bao_cao_Lab4.docx')
