#define CVDNN_USE

using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Dnn;
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
using static System.Runtime.InteropServices.JavaScript.JSType;
using Window = System.Windows.Window;


namespace MaskDetection
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
    {
		bool JIT_USE = true;
        VideoCapture m_capture;
		Thread t_cap;
        Mat m_mat;
        bool m_isRunning = false;
		bool b_Normalized = false;

		bool b_fullsize = false;
		bool b_facecog = true;

		OpenCvSharp.Dnn.Net net;
		OpenCvSharp.Size resz = new OpenCvSharp.Size(224, 224);

		// Mask_12K First
		static float[] mean = new float[3] { 0.5703f, 0.4665f, 0.4177f };
		static float[] std = new float[3] { 0.2429f, 0.2231f, 0.2191f };

		float face_confidence = 0.5f;

		public MainWindow()
		{
			InitializeComponent();
			m_capture = new VideoCapture();

			// Model Load (ONNX)
			var modelPath = "resnet18_Mask_EPOCH300_LR0.001_NormalTrue.onnx";
			net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(modelPath);
			slider_face_conf.Value = (int)(face_confidence * 100);
			check_facecog.IsChecked = b_facecog = true;
			if (check_facecog.IsChecked == true)
			{
				check_fullsize.IsEnabled = b_fullsize = false;
			}
		}
		private void Inference(Mat image, out int label, out double prob)
		{
			Mat resizedImage = image.Clone();

			Cv2.Resize(resizedImage, resizedImage, resz);
			Mat blob = new Mat();
			blob = CvDnn.BlobFromImage(resizedImage, 1/255.0f, 
				new OpenCvSharp.Size(224, 224),
				new OpenCvSharp.Scalar(mean[0] * 255, mean[1] * 255, mean[2] * 255), true, false);

			net.SetInput(blob);
			string[] outBlobNames = net.GetUnconnectedOutLayersNames();
			Mat[] outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();

			Mat matprob = net.Forward("output");
			double maxVal, minVal;
			OpenCvSharp.Point minLoc, maxLoc;
			Cv2.MinMaxLoc(matprob, out minVal, out maxVal, out minLoc, out maxLoc);
			label = maxLoc.X;
			prob = maxVal * 100.0f;
		}

		private void Btn_Click(object sender, RoutedEventArgs e)
		{
			if (sender.Equals(button_cam)) // 카메라 연결/해제 과정
			{
                if (!m_isRunning)
                {
					if (t_cap != null && t_cap.IsAlive)
					{
						MessageBox.Show("Camera is closing... Wait..", "Error");
						return;
					}
					t_cap = new Thread(new ThreadStart(Grab));
					t_cap.IsBackground = true; // 프로그램 꺼질 때 쓰레드도 같이 꺼짐
					m_isRunning = true;
					t_cap.Start();
					button_cam.Content = "Camera Close";
				}
                else {
					m_isRunning = false;
					image_cam.Source = null;
					button_cam.Content = "Camera Open";
				}
			}

			else if (sender.Equals(button_image)) // 이미지 로드 과정
			{

				OpenFileDialog openFileDialog = new OpenFileDialog();
				openFileDialog.Filter = "PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|JPG files (*.jpg)|*.jpg|JPEG files (*.jpeg)|*.jpeg|All files (*.*)|*.*";
				if (openFileDialog.ShowDialog() == true)
				{
					Mat image = Cv2.ImRead(openFileDialog.FileName);
					List<OpenCvSharp.Rect> faces;
					FaceCrop(image, out faces);
					int label = 0;
					double prob = 0.0f;

					for (int i = 0; i < faces.Count; i++)
					{
						OpenCvSharp.Rect bounds = new OpenCvSharp.Rect(0, 0, image.Cols, image.Rows);
						Mat roi = new Mat(image, faces[i] & bounds).Clone(); // cropped to fit image

						string str = "";
						Inference(image, out label, out prob);
						if (label == 0)
							Cv2.Rectangle(image, faces[i] & bounds, Scalar.Blue, 3);
						else
							Cv2.Rectangle(image, faces[i] & bounds, Scalar.Red, 3);
					}
					UI_Update(image, label, prob, true);
				}
			}

			else if (sender.Equals(check_facecog))
			{
				if (check_facecog.IsChecked == true)
				{
					b_facecog = true;
					check_fullsize.IsEnabled = b_fullsize = false;
				}
				else
				{
					b_facecog = false;
					check_fullsize.IsEnabled = b_fullsize = true;
				}
			}

			else if (sender.Equals(check_fullsize))
			{
				if (check_fullsize.IsChecked == true)
				{
					b_fullsize = false;
				}
				else
				{
					b_fullsize = true;
				}
			}
		}

		private void FaceCrop(Mat image, out List<OpenCvSharp.Rect> list)
		{
			list = new List<OpenCvSharp.Rect>();

			if (true) // SSD Model : 쓸만함
			{
				OpenCvSharp.Dnn.Net facenet;
				var prototext = "deploy.prototxt";
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
				if (data.Length == 1)
					return;
				for (int i = 0; i < rows; i++)
				{
					float confidence = data[i * cols + 2]; // Access confidence score

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

						// 그냥 face recognition 하면, 얼굴이 너무 빡세게 잡혀서.. 좌우상하 20% 정도씩 늘려줌.
						float face_scale_X = 0.2f;
						float face_scale_Y = 0.1f;
						if (x1 - (width * face_scale_X) < 0)
							x1 = 0;
						else
							x1 = x1 - (int)(width * face_scale_X);

						if (x2 + (width * face_scale_X) > image.Width)
							x2 = image.Width;
						else
							x2 = x2 + (int)(width * face_scale_X);


						if (y1 - (height * face_scale_Y) < 0)
							y1 = 0;
						else
							y1 = y1 - (int)(height * face_scale_Y);

						if (y2 + (height * face_scale_Y) > image.Height)
							y2 = image.Height;
						else
							y2 = y2 + (int)(height * face_scale_Y);


						OpenCvSharp.Rect item = new OpenCvSharp.Rect(x1, y1, x2-x1, y2-y1);
						Cv2.Rectangle(image,
							new OpenCvSharp.Point(item.Left, item.Top),
							new OpenCvSharp.Point(item.Right, item.Bottom),
							Scalar.Green, 2, LineTypes.Link8);

						list.Add(item);
					}
				}
			}

			if (false) // Cascade Classifier : 실사용하기에 매우 별로임
			{
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
        private void Grab() // 카메라를 연결하고 프레임을 읽고 추론까지 진행함
        {
			m_capture.Open(0, VideoCaptureAPIs.DSHOW);

			Mat frame = new Mat();
			while (m_isRunning)
			{
				if (m_capture.IsOpened() == true)
				{
					m_capture.Read(frame);
					if (!frame.Empty())
					{
						int label = 0;
						double prob = 0.0f;
						if (b_facecog == true)
						{
							List<OpenCvSharp.Rect> faces;
							FaceCrop(frame, out faces);

							for (int i = 0; i < faces.Count; i++)
							{
								OpenCvSharp.Rect bounds = new OpenCvSharp.Rect(0, 0, frame.Cols, frame.Rows);
								Mat roi = new Mat(frame, faces[i] & bounds).Clone(); // cropped to fit image

								string str = "";
								Inference(frame, out label, out prob);
								if (label == 0)
									Cv2.Rectangle(frame, faces[i] & bounds, Scalar.Blue, 3);
								else
									Cv2.Rectangle(frame, faces[i] & bounds, Scalar.Red, 3);
							}
						} else
						{
							Mat cropped = new Mat();
							cropped = frame.Clone();
							// 이미지 중앙의 400*400을 자르기 위함
							// 안자르면 얼굴 외 불필요한 배경들도 포함되어 정확도가 떨어짐
							if (b_fullsize == true) 
							{
								int crop_width = 400;
								int crop_height = 400;
								int center_x = frame.Width / 2, center_y = frame.Height / 2;
								int x1, y1, x2, y2;
								x1 = center_x - crop_width / 2;
								y1 = center_y - crop_height / 2;
								x2 = center_x + crop_width / 2;
								y2 = center_y + crop_height / 2;

								// 범위가 유효한지 확인 (이미지 크기를 벗어나지 않도록 제한)
								x1 = Math.Max(x1, 0);
								y1 = Math.Max(y1, 0);
								x2 = Math.Min(x2, frame.Width);
								y2 = Math.Min(y2, frame.Height);

								// 이미지 자르기
								OpenCvSharp.Rect roi = new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1);
								Cv2.Rectangle(frame, new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1), OpenCvSharp.Scalar.Red, 3);
								cropped = new Mat(frame, roi);
							}

							Inference(cropped, out label, out prob);
						}
						UI_Update(frame, label, prob, true);
					}
					Thread.Sleep(10); // prevent for lag
				} else
					m_isRunning = false;
			}
			if (m_capture.IsOpened())
				m_capture.Release();
		}

		private void UI_Update(Mat image, int label, double prob, bool bThread = false)
		{
			string str = "";
			if (bThread)
			{
				// UI
				image_cam.Dispatcher.Invoke(() =>
				{
					image_cam.Source = OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(image);
				});

				text_result.Dispatcher.Invoke(() =>
				{
					if (label == 0) // Mask
					{
						str = ":) Great! You're wearing a mask! (" + prob + "%)\n";
						text_result.Foreground = new SolidColorBrush(Colors.Blue);
					}
					else if (label == 1) // No Mask
					{
						str = ":( Put on your mask quickly! (" + prob + "%)\n";
						text_result.Foreground = new SolidColorBrush(Colors.Red);
					}
					text_result.Text = str;
				});
			}
			else
			{
				image_cam.Source = OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(image);

				if (label == 0) // Mask
				{
					str = ":) Great! You're wearing a mask! (" + prob + "%)\n";
					text_result.Foreground = new SolidColorBrush(Colors.Blue);
				}
				else if (label == 1) // No Mask
				{
					str = ":( Put on your mask quickly! (" + prob + "%)\n";
					text_result.Foreground = new SolidColorBrush(Colors.Red);
				}
				text_result.Text = str;
			}
		}
		private void ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
		{
			if (sender.Equals(slider_face_conf))
			{
				face_confidence = (float) slider_face_conf.Value / 100.0f;
			}
		}
	}
}