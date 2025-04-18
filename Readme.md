# TumorVision AI

An advanced medical imaging platform that combines deep learning-based tumor segmentation with GPT-4 Vision analysis for brain MRI scans.

## 🚀 Features

- Automated brain tumor segmentation using U-Net architecture
- Real-time analysis powered by GPT-4 Vision
- Interactive chat interface for medical queries
- Side-by-side visualization of original scans and segmentation results
- Structured medical reporting with anatomical context
- Support for multiple image formats (PNG, JPG, JPEG, TIF, TIFF)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Custom U-Net model (Tumor Segmentation)
  - GPT-4 Vision (Medical Analysis)
- **API**: Modal deployment
- **Image Processing**: PIL
- **Async Support**: aiohttp

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TumorVision.git
cd TumorVision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.streamlit/secrets.toml` file
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```

## 🚀 Running the Application

Launch the application using Streamlit:
```bash
streamlit run TumorVision.py
```

## 💻 Usage

1. Upload a brain MRI scan using the sidebar uploader
2. Wait for automatic segmentation and analysis
3. View the segmentation results and AI-generated medical analysis
4. Use the chat interface to ask specific questions about the scan
5. Clear the chat history using the "Clear Chat" button when needed

## 🔒 Security Notes

- API keys are managed securely through Streamlit secrets
- No permanent storage of medical images
- Stateless processing for enhanced privacy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- OpenAI for GPT-4 Vision API
- Modal for serverless deployment
- The medical imaging community


---

**Note**: This project is intended for research and educational purposes. Always consult qualified medical professionals for clinical decisions.