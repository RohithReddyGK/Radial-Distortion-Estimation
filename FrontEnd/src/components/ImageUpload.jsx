import React, { useRef } from "react";
import Lottie from "lottie-react";
import LoadingAnimation from "../assets/Loading.json";

export default function ImageUpload({ uploadedImage, setUploadedImage, checkerboard, setCheckerboard, onProcess, loading }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadedImage(file);
  };

  return (
    <div className="bg-white shadow-lg rounded-xl p-6 mb-6 max-w-xl mx-auto text-center">
      <h1 className="text-xl font-semibold mb-4">Upload Checkerboard Image</h1>
      <p className="mb-4 text-gray-600">Upload a single checkerboard or planar grid/floor image. For best results, use an oblique view.</p>

      <input type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef} className="mb-4" />

      {uploadedImage && (
        <div className="mb-4">
          <img src={URL.createObjectURL(uploadedImage)} alt="Preview" className="mx-auto rounded-lg border max-h-64 object-contain" />
        </div>
      )}

      <div className="flex justify-center gap-2 mb-4">
        <input
          type="number"
          min="1"
          value={checkerboard.cols}
          onChange={(e) => setCheckerboard({ ...checkerboard, cols: parseInt(e.target.value) })}
          className="border rounded px-2 py-1 w-20"
          placeholder="Cols"
        />
        <span className="text-gray-700 pt-1">x</span>
        <input
          type="number"
          min="1"
          value={checkerboard.rows}
          onChange={(e) => setCheckerboard({ ...checkerboard, rows: parseInt(e.target.value) })}
          className="border rounded px-2 py-1 w-20"
          placeholder="Rows"
        />
      </div>

      <button
        onClick={onProcess}
        className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 w-50 font-semibold"
      >
        {loading ? "Processing..." : "Process Image"}
      </button>

      {loading && (
        <div className="fixed inset-0 bg-white bg-opacity-85 flex items-center justify-center z-50">
          <Lottie animationData={LoadingAnimation} loop={true} className="w-64 h-64" />
        </div>
      )}
    </div>
  );
}
