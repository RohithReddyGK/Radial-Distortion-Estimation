import { useState, useRef } from "react";
import StepCard from "./components/StepCard";
import ImageUpload from "./components/ImageUpload";
import html2pdf from "html2pdf.js";

const BASE_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5000";

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [checkerboard, setCheckerboard] = useState({ cols: 7, rows: 5 });
  const [steps, setSteps] = useState([]);
  const [loading, setLoading] = useState(false);

  const stepsRef = useRef(null);

  const handleProcessImage = async () => {
    if (!uploadedImage) return alert("Please upload an image first!");
    setLoading(true);
    const formData = new FormData();
    formData.append("image", uploadedImage);
    formData.append("cols", checkerboard.cols);
    formData.append("rows", checkerboard.rows);

    try {
      const res = await fetch(`${BASE_URL}/api/process_image`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setSteps(data.steps);

      // Scroll to steps section after processing
      stepsRef.current?.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      console.error(err);
      alert("Backend not reachable!");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    const doc = document.getElementById("steps-container");
    const opt = {
      margin: 0.5,
      filename: "Camera_Calibration_Report.pdf",
      image: { type: "jpeg", quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: "in", format: "letter", orientation: "portrait" }
    };
    html2pdf().set(opt).from(doc).save();
  };

  return (
    <div className="relative min-h-screen w-full">
      <img
        src="/Background Image.png"
        alt="Computer Vision"
        className="absolute inset-0 w-full h-full object-cover"
      />
      <div className="relative z-10 min-h-screen p-6 font-sans flex flex-col items-center justify-center text-center">
        <h1 className="text-4xl text-black font-bold text-center mb-6">Computer Vision - Radial Distortion Estimation & Correction</h1>

        {/* Image Upload + Process Button inside card */}
        <ImageUpload
          uploadedImage={uploadedImage}
          setUploadedImage={setUploadedImage}
          checkerboard={checkerboard}
          setCheckerboard={setCheckerboard}
          onProcess={handleProcessImage}
          loading={loading}
        />

        {/* Steps */}
        <div ref={stepsRef} id="steps-container" className="space-y-6">
          {steps.map((step, idx) => (
            <StepCard key={idx} step={step} stepNumber={idx + 1} />
          ))}
        </div>

        {/* PDF Download + Run in Google Colab */}
        {steps.length > 0 && (
          <div className="flex justify-between gap-x-6 mt-6 max-w-4xl mx-auto">
            <button
              onClick={handleDownloadPDF}
              className="bg-green-600 text-white font-semibold px-4 py-2 rounded hover:bg-green-700 transition"
            >
              Download PDF Report
            </button>

            <a
              href="https://colab.research.google.com/drive/1k-9l6epFuxNLWOxhPtxD0w3ub1mNUSCq?usp=sharing"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-blue-600 text-white font-semibold px-4 py-2 rounded hover:bg-blue-700 transition"
            >
              Run in Google Colab
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
