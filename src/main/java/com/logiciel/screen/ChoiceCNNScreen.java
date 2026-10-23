/**
 * 
 */
package com.logiciel.screen;

import java.awt.Component;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class ChoiceCNNScreen extends JPanel{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	//
	public ChoiceCNNScreen(HomeScreen f) {
		// TODO Auto-generated constructor stub
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		add(Box.createVerticalStrut(30));
		//Title
		JLabel titre = new JLabel("Selectionnez le modèle de reconnaissance :");
		titre.setAlignmentX(Component.CENTER_ALIGNMENT); // Centrer le texte
		add(titre);
		add(Box.createVerticalStrut(40));
		//Buttons
		JButton btnPython = new JButton("100% Python Model (Interface dediee)");
		JButton btnDL4Java = new JButton("100% Java Model (DL4Java)");
		JButton btnONNX = new JButton("Python Imported Model (ONNX)");
		//
		btnPython.setAlignmentX(Component.CENTER_ALIGNMENT);
		btnDL4Java.setAlignmentX(Component.CENTER_ALIGNMENT);
		btnONNX.setAlignmentX(Component.CENTER_ALIGNMENT);
		//
		add(btnPython);
		add(Box.createVerticalStrut(15));
		add(btnDL4Java);
		add(Box.createVerticalStrut(15));
		add(btnONNX);
		//
		//Action python
		btnPython.addActionListener(e -> {
			System.out.println("Option choisie : Python Externe");
			//launch python script
			try {
				String path = "python/app_gui.py";
				ProcessBuilder pb = new ProcessBuilder("python",path);
				pb.inheritIO();
				pb.start();
				System.exit(0);
			}
			catch (Exception exc) {
				System.out.println("erreur" + exc);
			}
		});
		//Action DL4Java
		btnDL4Java.addActionListener(e -> {
			System.out.println("Option choisie : DL4Java");
			//DL4java + src screen
			f.displaySRCScreen("DL4Java");
		});
		//Action btnONNX
		btnONNX.addActionListener(e -> {
			System.out.println("Option choisie : ONNX");
			//ONNX + src screen
			f.displaySRCScreen("ONNX");
		});
	}
}
