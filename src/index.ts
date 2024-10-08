import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
  // @ts-ignore
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

let faceLandmarker: any;
let runningMode = "IMAGE";

const inputImageElement = document.getElementById("inputFile");
const canvasElement = document.getElementById("output") as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d");

async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1,
  });
}

async function loadImageToCanvas(file: File): Promise<HTMLImageElement> {
  const img = new Image();
  const objectURL = URL.createObjectURL(file);
  img.src = objectURL;

  return new Promise((resolve) => {
    img.onload = () => {
      canvasElement.width = img.width;
      canvasElement.height = img.height;
      canvasCtx!.drawImage(img, 0, 0);
      resolve(img);
    };
  });
}

// Função para calcular a distância entre dois pontos
function calculateDistance(point1: any, point2: any): number {
  return Math.sqrt(
    Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2)
  );
}

// Função para calcular yaw, pitch e roll baseado nos landmarks
function calculateHeadOrientation(landmarks: any): {
  yaw: number;
  pitch: number;
  roll: number;
} {
  const noseTip = landmarks[1]; // Landmarks para a ponta do nariz
  const rightEye = landmarks[33]; // Landmarks para o olho direito
  const leftEye = landmarks[263]; // Landmarks para o olho esquerdo
  const chin = landmarks[152]; // Landmarks para o queixo

  // Cálculo simplificado do yaw (direção horizontal do rosto)
  const yaw =
    Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) *
    (180 / Math.PI);

  // Cálculo do pitch (inclinação para cima ou para baixo)
  const eyeCenterY = (rightEye.y + leftEye.y) / 2;
  const pitch =
    Math.atan2(noseTip.y - eyeCenterY, chin.y - eyeCenterY) * (180 / Math.PI);

  // Cálculo do roll (rotação lateral)
  const roll =
    Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x) *
    (180 / Math.PI);

  return { yaw, pitch, roll };
}

// Função para calcular a razão da boca (largura vs. altura)
function calculateMouthRatio(landmarks: any): {
  width: number;
  height: number;
} {
  const leftMouthCorner = landmarks[61]; // Canto esquerdo da boca
  const rightMouthCorner = landmarks[291]; // Canto direito da boca
  const upperMouthCenter = landmarks[13]; // Centro do lábio superior
  const lowerMouthCenter = landmarks[14]; // Centro do lábio inferior

  const mouthWidth = calculateDistance(leftMouthCorner, rightMouthCorner);
  const mouthHeight = calculateDistance(upperMouthCenter, lowerMouthCenter);

  return { width: mouthWidth, height: mouthHeight };
}

function getFaceOrientation(
  yaw: number,
  pitch: number,
  roll: number,
  mouthWidth: number,
  mouthHeight: number
): string {
  const mouthRatio = mouthWidth / mouthHeight;

  let orientation = "";

  // Análise da direção do rosto com base no yaw
  if (yaw > 15) {
    orientation = "virado para a direita";
  } else if (yaw < -15) {
    orientation = "virado para a esquerda";
  } else {
    orientation = "de frente";
  }

  // Verificação adicional com base no pitch (inclinação vertical)
  if (pitch > 10) {
    orientation += " e inclinado para cima";
  } else if (pitch < -10) {
    orientation += " e inclinado para baixo";
  }

  // Análise da rotação com base no roll
  if (roll > 10) {
    orientation += " com rotação para a direita";
  } else if (roll < -10) {
    orientation += " com rotação para a esquerda";
  }

  // Ajustes adicionais baseados na proporção da boca
  if (mouthRatio > 1.8) {
    orientation += " (boca alargada, provavelmente de frente)";
  } else if (mouthRatio < 1.2) {
    orientation += " (boca estreita, provavelmente de lado)";
  }

  return orientation;
}

inputImageElement?.addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];

  if (file) {
    const img = await loadImageToCanvas(file);
    await createFaceLandmarker();

    const faceLandmarkerResult = await faceLandmarker.detect(img);

    const drawingUtils = new DrawingUtils(canvasCtx);

    for (const landmarks of faceLandmarkerResult.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        {
          color: "#E0E0E0",
        }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: "#30FF30" }
      );
    }

    const nose = faceLandmarkerResult.faceLandmarks[0][1];

    // Calcular a distância aproximada com base no valor z
    const zDistance = Math.abs(nose.z * 100); // Multiplicado por 100 para uma escala aproximada

    // Exibir a distância estimada
    // @ts-ignore
    document.getElementById(
      "distance"
    ).innerText = `Distância estimada do nariz: ${zDistance.toFixed(2)} in z`;

    const landmarks = faceLandmarkerResult.faceLandmarks[0];

    // Cálculo dos valores de yaw, pitch e roll
    const { yaw, pitch, roll } = calculateHeadOrientation(landmarks);

    // Cálculo da razão da boca
    const { width: mouthWidth, height: mouthHeight } =
      calculateMouthRatio(landmarks);

    // Obtenção da orientação com base nos valores
    const faceOrientation = getFaceOrientation(
      yaw,
      pitch,
      roll,
      mouthWidth,
      mouthHeight
    );

    document.getElementById(
      "orientation"
    )!.innerText = `Orientação do rosto: ${faceOrientation}`;
  }
});
