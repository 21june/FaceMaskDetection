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
		OpenCvSharp.Size resz = new OpenCvSharp.Size(224, 224); // Resnet input 224*224

		// Mask_12K
		static float[] mean = new float[3] { 0.5690f, 0.4653f, 0.4167f };
		static float[] std = new float[3] { 0.2425f, 0.2226f, 0.2186f };

		public MainWindow()
		{
			InitializeComponent();

			// For Camera
			m_capture = new VideoCapture();

			// Model Load (ONNX)
			var modelPath = "resnet18_Mask_EPOCH300_LR0.001_NormalTrue.onnx";
			net = OpenCvSharp.Dnn.CvDnn.ReadNetFromOnnx(modelPath);
		}
		private void Btn_Click(object sender, RoutedEventArgs e)
		{
			if (sender.Equals(button_cam)) // 카메라 연결/해제 과정
			{
				CameraOnOff();
            }

			else if (sender.Equals(button_image)) // 이미지 로드 과정
			{
				ImageLoadAndInference();
			}
		}

		private void ImageLoadAndInference()
		{
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|JPG files (*.jpg)|*.jpg|JPEG files (*.jpeg)|*.jpeg|All files (*.*)|*.*";
            if (openFileDialog.ShowDialog() == true)
            {
                Mat mat = Cv2.ImRead(openFileDialog.FileName);
                int label = 0;
                double prob = 0.0f;
                Inference(mat, out label, out prob, false);
                UI_Update(mat, label, prob);
            }

        }

		private void CameraOnOff()
		{
            if (!m_isRunning)
            {
                if (t_cap != null && t_cap.IsAlive)
                {
                    MessageBox.Show("Camera is closing... Wait..", "Error");
                    return;
                }
                t_cap = new Thread(new ThreadStart(GrabAndInference));
                t_cap.IsBackground = true; // 프로그램 꺼질 때 쓰레드도 같이 꺼짐
                m_isRunning = true;
                t_cap.Start();
                button_cam.Content = "Camera Close";
            }
            else
            {
                // 카메라 꺼지면 스레드 종료하고 이미지는 비워둠.
                m_isRunning = false;
                image_cam.Source = null;
                button_cam.Content = "Camera Open";
            }
        }

        private void GrabAndInference() // 카메라를 연결하고 프레임을 읽고 추론까지 진행함
        {
			m_capture.Open(0, VideoCaptureAPIs.DSHOW); // 카메라 연결

			Mat frame = new Mat();
			while (m_isRunning)
			{
				if (m_capture.IsOpened() == true)
				{
					m_capture.Read(frame); // 카메라 이미지 획득
					if (!frame.Empty())
					{
						int label = 0;
						double prob = 0.0f;
						Inference(frame, out label, out prob, true);
						UI_Update(frame, label, prob);
					}
					Thread.Sleep(10); // prevent for lag
				} else
					m_isRunning = false;
			}
			if (m_capture.IsOpened()) // 스레드 종료 시, 카메라가 연결되어있다면 해제.
				m_capture.Release();
		}

        private void Inference(Mat image, out int label, out double prob, bool bCrop = true)
        {
            Mat cropped = new Mat();
            image.CopyTo(cropped);

            // 이미지 중앙의 400*400을 자르기 위함
            // 안자르면 얼굴 외 불필요한 배경들도 포함되어 정확도가 떨어짐
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
            blob = CvDnn.BlobFromImage(cropped,
                1 / 255.0f, // scale factor로, 정규화를 위해 1/255로 곱해줌
                new OpenCvSharp.Size(224, 224),
                // mean의 값으로, mean을 먼저 뺀 다음에 scale factor로 곱하기 때문에, 정규화 되기 전 값인 0~255 범위로 바꾸어줘야함.
                new OpenCvSharp.Scalar(mean[0] * 255, mean[1] * 255, mean[2] * 255),
                true, false);
            net.SetInput(blob); // input 설정

            // 'output'의 값을 가져옴. 'output'은 onnx로 export 할 때 출력 이름으로 넣어준 값과 동일.
            Mat matprob = net.Forward("output");
            double maxVal, minVal;
            OpenCvSharp.Point minLoc, maxLoc;
            Cv2.MinMaxLoc(matprob, out minVal, out maxVal, out minLoc, out maxLoc); // 최대, 최소 값과 그 위치(라벨 인덱스)를 구함.
            label = maxLoc.X; // 최대 값의 라벨 인덱스를 구함. 여기서는 0이면 마스크 착용, 1이면 마스크 미착용.
            prob = maxVal * 100.0f; // 최대 값에 100을 곱해 퍼센테이지로 표현
        }


        private void UI_Update(Mat image, int label, double prob)
		{
			string str = "";
            // 메인 스레드에서는 이렇게 처리할 필요가 없지만 그 외 스레드에서는 UI 변경이 불가능 함.
            // 따라서 UI를 다루는 스레드에게 메시지큐를 보내 UI 변경 수행하는 방식을 사용. (Dispatcher.Invoke)
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
	}
}