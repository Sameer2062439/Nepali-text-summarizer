from rouge import Rouge

def calculate_rouge_scores(reference_summaries, generated_summaries):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    return scores

# Example usage
reference_summaries = ["भारत सरकारले प्रमुख विपक्षी दल कंग्रेसका अध्यक्ष राहुल गान्धीलाई उनीसँग विदेशी नागरिकता भए नभएको स्पष्ट पार्न भनेको छ।", "चार वर्षअघि नेपालमा आएको विनाशकारी भूकम्पपछि बन्द भएको नेपाल-चीन सीमामा रहेको तातोपानी नाका सञ्चालनमा आएपछि दुई देशबीचको व्यापारमा सकारात्मक प्रभाव पार्ने अपेक्षा व्यापारीहरूले राखेका छन्।"]
generated_summaries = ["भारतको गृह मन्त्रालयले पूर्वराष्ट्रपति र भारतीय जनता पार्टीका अध्यक्ष राहुल गान्धीलाई ब्रिटिश नागरिक घोषणा गरेको आरोपमा स्पष्टीकरण दिन निर्देशन दिएको छ", "भूकम्पपछि बन्द भएको तातोपानी नाकाको वैकल्पिक मार्ग थपिएपछि नेपालको बजार मूल्य ह्वात्तै घट्न सक्ने व्यवसायीहरूले बताएका छन्"]

rouge_scores = calculate_rouge_scores(reference_summaries, generated_summaries)
print(rouge_scores)
