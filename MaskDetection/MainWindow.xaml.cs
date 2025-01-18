#define CVDNN_USE

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Reflection.Emit;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;
using static System.Collections.Specialized.BitVector32;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Window = System.Windows.Window;
using System.Runtime.InteropServices;


namespace MaskDetection
{
	public partial class MainWindow : Window
	{
		VideoCapture m_capture;
		Thread t_cap;
		bool m_isRunning = false;
		bool b_facedet = true;

		OpenCvSharp.Dnn.Net net;
		OpenCvSharp.Size resz = new OpenCvSharp.Size(224, 224);

		// Mean & Standard Deviation
		// mean, std 구해놨지만 안쓰는게 더 잘 나옴
		bool b_meanstd = false;
		static float[] mean = new float[3] { 0.5703f, 0.4665f, 0.4177f };
		static float[] std = new float[3] { 0.2429f, 0.2231f, 0.2191f };

		float face_confidence = 0.3f;

		public MainWindow()
		{
			InitializeComponent();
			m_capture = new VideoCapture();

			// Download: https://github.com/21june/PyroNote/blob/main/MaskDetection/resnet18_Mask_12K_None_EPOCH200_LR0.0001.onnx
			string mask_model = "resnet18_Mask_12K_None_EPOCH200_LR0.0001.onnx";
			net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(mask_model);

			// UI 처리
			if (!net.Empty())	text_model.Text = "Model: " + mask_model;
			else				text_model.Text = "Model: " + "No Model";

			slider_face_conf.Value = (int)(face_confidence * 100);
			check_facedet.IsChecked = b_facedet = true;
		}

		private void ThreadFunc() // 카메라를 연결하고 프레임을 읽고 추론까지 진행함
		{
			m_capture.Open(0, VideoCaptureAPIs.DSHOW);

			Mat frame = new Mat();
			while (m_isRunning)
			{
				if (m_capture.IsOpened() == true)
				{
					m_capture.Read(frame);
					if (!frame.Empty())
						Run(frame);
					Thread.Sleep(10); // prevent for lag
				}
				else
				{
					m_isRunning = false;
					image_cam.Dispatcher.Invoke(() => { image_cam.Source = null; });
					button_cam.Dispatcher.Invoke(() => { button_cam.Content = "Open"; });
				}
			}
			if (m_capture.IsOpened())
				m_capture.Release();
		}

		private void Run(Mat image)
		{
			List<OpenCvSharp.Rect> faces;
			ImageROI(image, out faces);

			int label = 0;
			double prob = 0.0f;
			for (int i = 0; i < faces.Count; i++)
			{
				OpenCvSharp.Rect bounds = new OpenCvSharp.Rect(0, 0, image.Cols, image.Rows);
				OpenCvSharp.Rect rt = faces[i] & bounds;
				Mat roi = new Mat(image, rt).Clone(); // cropped to fit image
				string str = "";
				Inference(roi, out label, out prob);
				if (label == 0)
				{
					Cv2.Rectangle(image, rt, Scalar.Blue, 3);
					Cv2.PutText(image, prob.ToString("0.00"), new OpenCvSharp.Point(rt.X, rt.Y), HersheyFonts.HersheyDuplex, 1.0, Scalar.Black);
				}
				else if (label == 1)
				{
					Cv2.Rectangle(image, rt, Scalar.Red, 3);
					Cv2.PutText(image, prob.ToString("0.00"), new OpenCvSharp.Point(rt.X, rt.Y), HersheyFonts.HersheyDuplex, 1.0, Scalar.Black);
				}
				else
				{
					Cv2.Rectangle(image, rt, Scalar.Gray, 3);
					Cv2.PutText(image, "N/A", new OpenCvSharp.Point(rt.X, rt.Y), HersheyFonts.HersheyDuplex, 1.0, Scalar.Black);
				}
			}

			image_cam.Dispatcher.Invoke(() => { image_cam.Source = OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(image); });
		}

		private void ImageROI(Mat image,  out List<OpenCvSharp.Rect> faces)
		{
			faces = new List<OpenCvSharp.Rect>();

			if (b_facedet == true)
			{
				FaceCrop(image, out faces);
			}
			else
			{
				OpenCvSharp.Rect rt = new OpenCvSharp.Rect(0, 0, image.Width, image.Height);
				faces.Add(rt);
			}

		}


		// face detection for using ssd (or haarcascade)
		private void FaceCrop(Mat image, out List<OpenCvSharp.Rect> list)
		{
			list = new List<OpenCvSharp.Rect>();

			if (true) // SSD Model : Useful
			{
				OpenCvSharp.Dnn.Net facenet;
				// Download: https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/deploy.prototxt
				var prototext = "deploy.prototxt";
				// Download: https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel
				var modelPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
				facenet = Net.ReadNetFromCaffe(prototext, modelPath);
				Mat inputBlob = CvDnn.BlobFromImage(
					image, 1, new OpenCvSharp.Size(300, 300), new OpenCvSharp.Scalar(104, 177, 123),
					false, false
				);
				facenet.SetInput(inputBlob, "data");
				string[] outputs = facenet.GetUnconnectedOutLayersNames();
				Mat outputBlobs = facenet.Forward("detection_out");
				Mat ch1Blobs = outputBlobs.Reshape(1, 1);

				int rows = outputBlobs.Size(2);
				int cols = outputBlobs.Size(3);
				long total = outputBlobs.Total();
				ch1Blobs.GetArray(out float[] data);
				if (data.Length == 1) return;

				for (int i = 0; i < rows; i++)
				{
					float confidence = data[i * cols + 2]; // Access confidence score

					// 설정된 confidence 값보다 클 경우만
					if (confidence > face_confidence)
					{
						int x1 = (int)(data[i * cols + 3] * image.Width);
						int y1 = (int)(data[i * cols + 4] * image.Height);
						int x2 = (int)(data[i * cols + 5] * image.Width);
						int y2 = (int)(data[i * cols + 6] * image.Height);

						OpenCvSharp.Rect rt = new OpenCvSharp.Rect(x1, y1, x2, y2);

						int centerX = (rt.Left + rt.Right) / 2;
						int centerY = (rt.Top + rt.Bottom) / 2;

						int width = x2 - x1;
						int height = y2 - y1;

						// 그냥 face recognition 하면, 얼굴이 너무 빡세게 잡혀서.. 가로세로 10% 정도씩 늘려줌.
						float face_scale_X = 0.1f;
						float face_scale_Y = 0.1f;
						if (x1 - (width * face_scale_X) < 0) x1 = 0;
						else x1 = x1 - (int)(width * face_scale_X);

						if (x2 + (width * face_scale_X) > image.Width) x2 = image.Width;
						else x2 = x2 + (int)(width * face_scale_X);

						if (y1 - (height * face_scale_Y) < 0) y1 = 0;
						else y1 = y1 - (int)(height * face_scale_Y);

						if (y2 + (height * face_scale_Y) > image.Height) y2 = image.Height;
						else y2 = y2 + (int)(height * face_scale_Y);

						OpenCvSharp.Rect item = new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1);
						list.Add(item);
					}
				}
			}

			if (false) // Cascade Classifier : Useless
			{
				// Download: https://github.com/mitre/biqt-face/tree/master/config/haarcascades
				string filenameFaceCascade = "haarcascade_frontalface_alt2.xml";
				CascadeClassifier faceCascade = new CascadeClassifier();
				if (!faceCascade.Load(filenameFaceCascade))
				{
					Console.WriteLine("error");
					return;
				}

				// detect 
				OpenCvSharp.Rect[] faces = faceCascade.DetectMultiScale(image);
				foreach (var item in faces)
				{
					list.Add(item);
					Cv2.Rectangle(image, item, Scalar.Red); // add rectangle to the image
					Console.WriteLine("faces : " + item);
				}
			}
		}

		// 정규화 과정
		private void NormalizeImage(ref Mat img)
		{
			img.ConvertTo(img, MatType.CV_32FC3);
			Mat[] rgb = img.Split();

			// 0.0f~1.0f
			rgb[0] = rgb[0].Divide(255.0f); // B
			rgb[1] = rgb[1].Divide(255.0f); // G
			rgb[2] = rgb[2].Divide(255.0f); // R
			if (b_meanstd)
			{
				// mean
				rgb[2] = rgb[2].Subtract(new Scalar(mean[0])); // B
				rgb[1] = rgb[1].Subtract(new Scalar(mean[1])); // G
				rgb[0] = rgb[0].Subtract(new Scalar(mean[2])); // R

				// std
				rgb[2] = rgb[2].Divide(std[0]); // B
				rgb[1] = rgb[1].Divide(std[1]); // G
				rgb[0] = rgb[0].Divide(std[2]); // R
			}
			Cv2.Merge(rgb, img);
			Cv2.Resize(img, img, resz);
		}

		// 추론 과정
		private void Inference(Mat image, out int label, out double prob)
		{
			if (net.Empty())
			{
				MessageBox.Show("No Found Model!");
				label = -1; prob = 0;
				return;
			}
			if (image.Empty()) {
				label = -1; prob = 0; return;
			}
			Mat resizedImage = image.Clone();
			Mat blob = new Mat();
			NormalizeImage(ref resizedImage); 
			blob = CvDnn.BlobFromImage(resizedImage, 1.0f,
				new OpenCvSharp.Size(224, 224), swapRB:true, crop:false);

			net.SetInput(blob);
			string[] outBlobNames = net.GetUnconnectedOutLayersNames();
			Mat[] outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();
			Mat matprob = net.Forward("output");

			// 최대 값의 구하기
			double maxVal, minVal;
			OpenCvSharp.Point minLoc, maxLoc;
			Cv2.MinMaxLoc(matprob, out minVal, out maxVal, out minLoc, out maxLoc);
			label = maxLoc.X;
			prob = maxVal * 100;
		}

		// 버튼 클릭 이벤트
		private void ClickEvent(object sender, RoutedEventArgs e)
		{
			if (sender.Equals(button_cam)) // 카메라 연결/해제 버튼
			{
				if (!m_isRunning)
				{
					if (t_cap != null && t_cap.IsAlive)
					{
						MessageBox.Show("Camera is closing... Wait..", "Error");
						return;
					}
					t_cap = new Thread(new ThreadStart(ThreadFunc));
					t_cap.IsBackground = true; // 프로그램 꺼질 때 쓰레드도 같이 꺼짐
					m_isRunning = true;
					t_cap.Start();
					button_cam.Content = "Close";
				}
				else
				{
					m_isRunning = false;
					image_cam.Source = null;
					button_cam.Content = "Open";
				}
			}

			else if (sender.Equals(button_image)) // 이미지 로드 버튼
			{

				OpenFileDialog openFileDialog = new OpenFileDialog();
				openFileDialog.Filter = "PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|JPG files (*.jpg)|*.jpg|JPEG files (*.jpeg)|*.jpeg|All files (*.*)|*.*";
				if (openFileDialog.ShowDialog() == true)
				{
					Mat image = Cv2.ImRead(openFileDialog.FileName);
					if (!image.Empty()) text_image.Text = "Image: " + openFileDialog.SafeFileName;
					else text_image.Text = "Image: " + "No Model";
					Run(image);
				}
			}
			else if (sender.Equals(button_model)) // 모델 로드 버튼
			{

				OpenFileDialog openFileDialog = new OpenFileDialog();
				openFileDialog.Filter = "ONNX Weight files (*.onnx)|*.onnx|All files (*.*)|*.*"; // ONNX 파일만
				if (openFileDialog.ShowDialog() == true)
				{
					net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(openFileDialog.FileName);
					if (!net.Empty())	text_model.Text = "Model: " + openFileDialog.SafeFileName;
					else				text_model.Text = "Model: " + "No Model";
				}
			}
			else if (sender.Equals(check_facedet)) // face detection 체크 박스
			{
				if (check_facedet.IsChecked == true)	b_facedet = true;
				else									b_facedet = false;
			}
		}

		// Slider 값 변경 이벤트
		private void SliderEvent(object sender, RoutedPropertyChangedEventArgs<double> e)
		{
			if (sender.Equals(slider_face_conf))
			{
				face_confidence = (float)slider_face_conf.Value / 100.0f;
			}
		}
	}
}