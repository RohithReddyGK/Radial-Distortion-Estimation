const StepCard = ({ step, stepNumber }) => {
  return (
    <div className="max-w-4xl mx-auto bg-gray-200 p-6 rounded-xl shadow-md">
      <h2 className="font-semibold text-xl mb-2">
        {step.title}
      </h2>

      {step.description && <p className="text-gray-700 mb-2 font-medium whitespace-pre-line">{step.description}</p>}
      {step.explanation && <p className="text-gray-600 mb-4 italic whitespace-pre-line">{step.explanation}</p>}

      {/* Table */}
      {step.table && step.table.length > 0 && (
        <div className="overflow-x-auto mb-4">
          <table className="table-auto border-collapse border border-gray-300 w-full">
            <thead>
              <tr className="bg-gray-100">
                {Object.keys(step.table[0]).map((key) => (
                  <th key={key} className="border border-gray-300 px-2 py-1 text-left text-sm">{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {step.table.map((row, idx) => (
                <tr key={idx}>
                  {Object.values(row).map((val, i) => (
                    <td key={i} className="border border-gray-300 px-2 py-1 text-sm">
                      {Array.isArray(val) ? (
                        <pre className="whitespace-pre-wrap">
                          {val.map(r => Array.isArray(r) ? r.join('   ') : r).join('\n')}
                        </pre>
                      ) : (
                        val
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>

          </table>
        </div>
      )}

      {/* Images */}
      {step.imagesBase64 && step.imagesBase64.length > 0 && (
        <div className="gap-4 mt-2 flex flex-wrap justify-center">
          {step.imagesBase64.map((img, idx) => (
            <img key={idx} src={img} alt={`Step ${stepNumber} Image ${idx + 1}`} className="rounded-md border border-gray-200 w-full max-w-md" />
          ))}
        </div>
      )}
    </div>
  );
};

export default StepCard;
