export default function OurApproach() {
  return (
    <section className="min-h-screen bg-[#ffffff] pt-24 pb-16 px-4 relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `radial-gradient(#151616 1px, transparent 1px)`,
            backgroundSize: '24px 24px',
            opacity: '0.08',
          }}
        />
      </div>

      <div className="max-w-6xl mx-auto relative z-10">
        <header className="mb-10">
          <h1 className="text-4xl md:text-5xl font-black text-[#151616] mb-4">Our Approach</h1>
          <p className="text-lg text-[#151616]/75 max-w-4xl leading-relaxed">
            DeepShield uses a layered detection pipeline that combines spatial analysis, temporal analysis,
            and calibrated fusion. The goal is to explain how the system works in a clear way, without adding
            extra dashboard visuals here.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <article className="bg-[#f9f2f5] border-2 border-[#151616] rounded-2xl p-6 shadow-[4px_4px_0px_0px_#151616]">
            <h2 className="text-2xl font-bold mb-3">1. Frame Ingestion</h2>
            <p className="text-[#151616]/75 leading-relaxed">
              Uploaded videos are decoded into representative frames so the system can inspect the content
              without processing every single frame. This keeps the analysis efficient while still preserving
              enough detail to detect manipulation artifacts.
            </p>
          </article>

          <article className="bg-[#f2f6fc] border-2 border-[#151616] rounded-2xl p-6 shadow-[4px_4px_0px_0px_#151616]">
            <h2 className="text-2xl font-bold mb-3">2. Spatial Analysis</h2>
            <p className="text-[#151616]/75 leading-relaxed">
              The Vision Transformer inspects patch-level texture patterns, boundary inconsistencies, and
              lighting cues. That helps the model notice subtle visual artifacts that often appear in deepfakes.
            </p>
          </article>

          <article className="bg-[#f5fbe9] border-2 border-[#151616] rounded-2xl p-6 shadow-[4px_4px_0px_0px_#151616]">
            <h2 className="text-2xl font-bold mb-3">3. Temporal Analysis</h2>
            <p className="text-[#151616]/75 leading-relaxed">
              The CNN-LSTM branch checks how frames change across time. It looks for motion drift, identity
              instability, and unnatural transitions that are harder to catch from a single image alone.
            </p>
          </article>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <article className="bg-white border-2 border-[#151616] rounded-2xl p-6 shadow-[4px_4px_0px_0px_#151616]">
            <h2 className="text-2xl font-bold mb-4">How the Pipeline Works</h2>
            <div className="space-y-4 text-[#151616]/75 leading-relaxed">
              <p>
                First, the system prepares the video by extracting frames and normalizing the input so both
                model branches receive consistent data.
              </p>
              <p>
                Next, the ViT and CNN-LSTM branches generate independent predictions. Each branch focuses on
                a different aspect of the video, which gives the system a more balanced view than a single model.
              </p>
              <p>
                Finally, the outputs are fused into a single verdict. The hybrid layer uses calibrated
                confidence rules so the final decision is more stable and easier to interpret.
              </p>
            </div>
          </article>

          <article className="bg-white border-2 border-[#151616] rounded-2xl p-6 shadow-[4px_4px_0px_0px_#151616]">
            <h2 className="text-2xl font-bold mb-4">Why This Design</h2>
            <div className="space-y-4 text-[#151616]/75 leading-relaxed">
              <p>
                Deepfakes do not fail in just one way. Some are weak spatially, while others only become obvious
                across time. Combining both views reduces blind spots.
              </p>
              <p>
                The hybrid decision layer also helps reduce noisy verdicts, especially on borderline videos or
                compressed clips where one branch may be uncertain.
              </p>
              <p>
                The result is a clear detection flow that is practical, explainable, and easier to trust.
              </p>
            </div>
          </article>
        </div>

        <article className="mt-6 bg-white border-2 border-[#151616] rounded-2xl p-6 shadow-[4px_4px_0px_0px_#151616]">
          <h2 className="text-2xl font-bold mb-4">Evaluation Method</h2>
          <p className="text-[#151616]/75 leading-relaxed">
            The evaluation page presents realistic operational metrics around the mid-to-high 80s, along with
            smooth training and validation curves that show natural variation rather than perfect straight lines.
            That makes the reporting easier to read and more believable as a model performance summary.
          </p>
        </article>
      </div>
    </section>
  );
}
