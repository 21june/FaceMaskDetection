#define CVDNN_USE

using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.ComponentModel;
using System.IO;
using System.Reflection.Emit;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
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

		OpenCvSharp.Dnn.Net net;
		OpenCvSharp.Size resz = new OpenCvSharp.Size(224, 224);

		// Mask_12K First
		static float[] mean = new float[3] { 0.5690f, 0.4653f, 0.4167f };
		static float[] std = new float[3] { 0.2425f, 0.2226f, 0.2186f };

		public MainWindow()
		{
			InitializeComponent();
			m_capture = new VideoCapture();

			// Model Load (ONNX)
			var modelPath = "resnet18_Mask_EPOCH300_LR0.001_NormalTrue.onnx";
			net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(modelPath);
		}
		private void Inference(Mat image, out int label, out double prob, bool bCrop = true)
		{
			Mat cropped = new Mat();
			image.CopyTo(cropped);
			if (bCrop)
			{
				int crop_width = 400;
				int crop_height = 400;
				int center_x = image.Width / 2, center_y = image.Height / 2;
				int x1, y1, x2, y2;
				x1 = center_x - crop_width / 2;
				y1 = center_y - crop_height / 2;
				x2 = center_x + crop_width / 2;
				y2 = center_y + crop_height / 2;

				// 범위가 유효한지 확인 (이미지 크기를 벗어나지 않도록 제한)
				x1 = Math.Max(x1, 0);
				y1 = Math.Max(y1, 0);
				x2 = Math.Min(x2, image.Width);
				y2 = Math.Min(y2, image.Height);
				// 이미지 자르기
				OpenCvSharp.Rect roi = new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1);
				Cv2.Rectangle(image, new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1), OpenCvSharp.Scalar.Red, 3);
				cropped = new Mat(image, roi);
			}

			Cv2.Resize(cropped, cropped, resz);
			Mat blob = new Mat();
			blob = CvDnn.BlobFromImage(cropped, 1/255.0f, 
				new OpenCvSharp.Size(224, 224),
				new OpenCvSharp.Scalar(145, 119, 106), true, false);

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
					Mat mat = Cv2.ImRead(openFileDialog.FileName);
					int label = 0;
					double prob = 0.0f;
					string str = "";
					Inference(mat, out label, out prob, false);
					UI_Update(mat, label, prob, true);
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
						string str = "";
						Inference(frame, out label, out prob, true);
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
	}
}