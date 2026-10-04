from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Original bootstrap knowledge written for the prototype. Each item becomes
# several short training records so the model sees both Hindi and English.
KB = [
    ("गेहूं", "wheat", "रबी", "nitrogen", "पीली", "सिंचाई"),
    ("धान", "rice", "खरीफ", "nitrogen", "भूरी", "जल प्रबंधन"),
    ("सरसों", "mustard", "रबी", "sulfur", "पीली", "सिंचाई"),
    ("आलू", "potato", "रबी", "potassium", "भूरी", "जल निकास"),
    ("टमाटर", "tomato", "खरीफ", "nitrogen", "पीली", "सिंचाई"),
    ("कपास", "cotton", "खरीफ", "nitrogen", "पीली", "जल प्रबंधन"),
    ("मक्का", "maize", "खरीफ", "nitrogen", "पीली", "सिंचाई"),
    ("चना", "chickpea", "रबी", "phosphorus", "पीली", "जल निकास"),
]

base = []
for hi, en, season, nutrient, symptom, water in KB:
    base.extend([
        f"{hi} एक {season} फसल है। {en} is an important crop in Indian agriculture.",
        f"{hi} की अच्छी वृद्धि के लिए मिट्टी की जांच करना उपयोगी है।",
        f"Soil testing helps decide fertilizer and nutrient management for {en}.",
        f"{hi} की पत्तियां {symptom} होने पर पोषक तत्वों की कमी, पानी की समस्या या रोग जैसे कई कारणों की जांच करनी चाहिए।",
        f"Yellow or {symptom} leaves do not prove a single cause; inspect roots, soil moisture and disease symptoms.",
        f"{hi} में {water} सही रखना फसल के स्वास्थ्य के लिए महत्वपूर्ण है।",
        f"Good irrigation scheduling avoids both drought stress and unnecessary standing water in the field.",
        f"किसान को खाद की मात्रा तय करने से पहले मिट्टी, फसल की अवस्था और स्थानीय कृषि सलाह देखनी चाहिए।",
        f"A farmer should consider soil test results, crop stage, weather and local extension advice before applying fertilizer.",
        f"कीट या रोग की पहचान केवल एक लक्षण देखकर नहीं करनी चाहिए। कई लक्षण अलग समस्याओं में समान दिखाई दे सकते हैं।",
        f"Pest and disease identification should use multiple symptoms, the crop stage and the field context.",
    ])

# Agriculture concepts useful for a conversational prototype.
extra = [
    "मृदा परीक्षण से pH, उपलब्ध पोषक तत्व और उर्वरक प्रबंधन के बारे में जानकारी मिल सकती है।",
    "Soil pH affects nutrient availability and can change the response of a crop to fertilizer.",
    "नाइट्रोजन पौधों की पत्तियों और वनस्पतिक वृद्धि में महत्वपूर्ण भूमिका निभाता है।",
    "Nitrogen is an essential plant nutrient associated with vegetative growth and leaf development.",
    "फास्फोरस जड़ों और पौधों की ऊर्जा संबंधी प्रक्रियाओं के लिए आवश्यक पोषक तत्व है।",
    "Potassium is involved in water regulation and several physiological processes in plants.",
    "ड्रिप सिंचाई पानी को फसल की जड़ क्षेत्र के पास देने का एक तरीका है।",
    "Drip irrigation can deliver water near the root zone and may reduce losses when properly designed.",
    "मल्चिंग मिट्टी की सतह को ढककर नमी और खरपतवार प्रबंधन में मदद कर सकती है।",
    "Mulching can help with soil moisture conservation and weed management.",
    "फसल चक्र बदलने से खेत में एक ही फसल के लगातार उगने से जुड़े कुछ दबाव कम किए जा सकते हैं।",
    "Crop rotation can diversify the sequence of crops grown on a field and can support soil and pest management.",
    "पत्तियों पर धब्बे कई कारणों से हो सकते हैं, इसलिए फोटो, मौसम और खेत का इतिहास उपयोगी जानकारी है।",
    "Leaf spots can have multiple causes, so crop history, weather and image evidence should be considered together.",
    "तेज बारिश के बाद खेत में जलभराव होने पर जड़ों के आसपास ऑक्सीजन कम हो सकती है।",
    "After heavy rain, prolonged waterlogging can reduce oxygen around roots.",
    "रोग प्रबंधन में स्वच्छ रोपण सामग्री, खेत की निगरानी और जरूरत के अनुसार नियंत्रण उपाय महत्वपूर्ण हैं।",
    "Integrated pest management combines monitoring, prevention and appropriate control measures.",
    "उर्वरक की सटीक मात्रा स्थानीय मिट्टी और फसल की सिफारिश पर निर्भर करती है।",
    "Fertilizer dose should be based on crop need, soil test information and locally recommended practice.",
    "किसान की सलाह देते समय स्थान, मौसम, फसल की अवस्था और उपलब्ध संसाधनों को ध्यान में रखना चाहिए।",
    "Agricultural advice should consider location, weather, crop stage and available farm resources.",
]
base.extend(extra * 12)

# Make the corpus less order-dependent by repeating and shuffling short records.
random.seed(42)
random.shuffle(base)
train_records = []
for i, line in enumerate(base * 12):
    line = line.strip()
    train_records.append(f"### agriculture_record_{i}\n{line}\n")

(DATA / "labeled_corpus.txt").write_text("\n".join(train_records), encoding="utf-8")

unlabeled = [
    "गेहूं की पत्तियां पीली हैं, क्या कारण हो सकता है?",
    "धान में पत्तियों पर भूरे धब्बे दिखाई दे रहे हैं।",
    "सरसों में पत्तियां पीली हो रही हैं।",
    "आलू के खेत में बारिश के बाद पानी जमा है।",
    "टमाटर की पत्तियों पर धब्बे क्यों हैं?",
    "मक्का में पौधे कमजोर दिख रहे हैं।",
    "चना की फसल के लिए मिट्टी की जांच क्यों करें?",
    "How can soil testing help a farmer?",
    "What can cause yellow leaves in a crop?",
    "Why is irrigation scheduling important?",
    "What should a farmer check before applying fertilizer?",
    "How can waterlogging affect roots?",
    "What information is useful when identifying a crop disease?",
    "Explain integrated pest management simply.",
    "What does nitrogen do for plants?",
    "How does potassium help plants?",
    "What is drip irrigation?",
    "Why can crop rotation be useful?",
    "What should I do when I see leaf spots?",
    "मेरी फसल में कीड़े दिख रहे हैं, पहले क्या जांचूं?",
]
(DATA / "unlabeled_queries.txt").write_text("\n".join(unlabeled) + "\n", encoding="utf-8")
print(f"Wrote {len(train_records)} training records to {DATA/'labeled_corpus.txt'}")
print(f"Wrote {len(unlabeled)} unlabeled queries to {DATA/'unlabeled_queries.txt'}")
