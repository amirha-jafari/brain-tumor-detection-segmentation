# 🧠 Brain Tumor Detection & Segmentation

سیستمی مبتنی بر Deep Learning و Computer Vision برای تحلیل تصاویر MRI مغز — تشخیص وجود تومور و مشخص‌کردن دقیق ناحیه‌ی آن به‌صورت pixel-level.

> ⚠️ **این پروژه صرفاً برای اهداف پورتفولیو و یادگیری ساخته شده و برای استفاده‌ی بالینی یا تشخیص پزشکی طراحی نشده است.**

## Overview

سیستم پس از دریافت یک تصویر MRI:
1. تشخیص می‌دهد که آیا تومور وجود دارد یا خیر (Classification)
2. در صورت وجود تومور، ناحیه‌ی دقیق آن را با یک Segmentation Mask مشخص می‌کند
3. نتیجه را همراه با میزان اطمینان مدل (confidence) ارائه می‌دهد

## Status

🚧 در حال توسعه — این پروژه فاز به فاز ساخته می‌شود. وضعیت فعلی: **فاز ۱ (Setup) تکمیل شده**

## Tech Stack

- Python, PyTorch, torchvision
- segmentation-models-pytorch (U-Net)
- OpenCV, Albumentations
- Streamlit (دمو)

## Project Structure

```
brain-tumor-detection-segmentation/
├── config/          # تنظیمات پروژه
├── data/            # داده (خام و پردازش‌شده)
├── models/          # وزن مدل‌های آموزش‌دیده
├── notebooks/       # اکتشاف و آزمایش
├── src/             # کد اصلی پروژه
│   ├── data/            # dataset و preprocessing
│   ├── models/          # معماری مدل‌ها
│   ├── training/        # اسکریپت‌های train
│   ├── evaluation/      # متریک‌ها
│   └── inference/       # پایپ‌لاین پیش‌بینی
├── outputs/         # نتایج و نمودارها
├── tests/           # تست‌ها
├── app.py           # رابط کاربری دمو
└── main.py          # نقطه‌ی ورود اصلی
```

## Setup

```bash
git clone https://github.com/amirha-jafari/brain-tumor-detection-segmentation.git
cd brain-tumor-detection-segmentation
python -m venv venv
source venv/bin/activate   # ویندوز: venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap

- [x] Setup و اسکلت پروژه
- [ ] بررسی و آماده‌سازی دیتاست
- [ ] مدل Classification اولیه
- [ ] ارزیابی Classification
- [ ] مدل Segmentation با U-Net
- [ ] ارزیابی Segmentation
- [ ] ترکیب Pipeline کامل
- [ ] رابط کاربری + دمو زنده
- [ ] مستندسازی نهایی

## License

MIT
