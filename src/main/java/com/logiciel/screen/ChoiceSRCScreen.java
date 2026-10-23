/**
 * 
 */
package com.logiciel.screen;

import java.awt.Component;
import java.io.File;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import com.logiciel.engine.EngineImage;
import com.logiciel.engine.EngineVideo;
import com.logiciel.engine.EngineWebcam;


/**
 * 
 */
public class ChoiceSRCScreen extends JPanel{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private String modelChoice; //CNN 2 value possible {"ONNX","DL4Java"}
	/**
	 * 
	 */
	public ChoiceSRCScreen(HomeScreen f,String modelChoice) {
		// TODO Auto-generated constructor stub
		this.modelChoice = modelChoice;
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		add(Box.createVerticalStrut(30));
		//Title
		JLabel title = new JLabel("Mode choisi : " + modelChoice + ". Choisissez la source vidéo :");
		title.setAlignmentX(Component.CENTER_ALIGNMENT);
		add(title);
		add(Box.createVerticalStrut(40));
		//buttons
		JButton btnImage = new JButton("Image enregistrée");
		JButton btnVideo = new JButton("Vidéo enregistrée");
		JButton btnWebcam = new JButton("Temps réel (Webcam)");
		JButton btnRetour = new JButton("<- Retour");
		//buttons settings
		btnImage.setAlignmentX(Component.CENTER_ALIGNMENT);
		btnVideo.setAlignmentX(Component.CENTER_ALIGNMENT);
		btnWebcam.setAlignmentX(Component.CENTER_ALIGNMENT);
		btnRetour.setAlignmentX(Component.CENTER_ALIGNMENT);
		//
		add(btnImage);
		add(Box.createVerticalStrut(15));
		add(btnVideo);
		add(Box.createVerticalStrut(15));
		add(btnWebcam);
		add(Box.createVerticalStrut(40)); // Plus grand espace avant le bouton retour
		add(btnRetour);
		//Action Listener
		btnImage.addActionListener(e -> {
			System.out.println("Source : Image pour le CNN " + this.modelChoice);
			JFileChooser exploreur = new JFileChooser();
			//
			int resultat = exploreur.showOpenDialog(this);
			if (resultat == JFileChooser.APPROVE_OPTION) {
				File file = exploreur.getSelectedFile();
				String path = file.getAbsolutePath();
				//
				try {
					EngineImage.executer(path,this.modelChoice);
				}
				catch (Exception exc) {
					JOptionPane.showMessageDialog(this, "Error Image Launch: " + exc.getMessage());
				}
			}
		});
		//
		btnVideo.addActionListener(e -> {
			// C'est ici qu'on ouvrira le JFileChooser pour sélectionner une image
			System.out.println("Source : video pour le CNN " + this.modelChoice);
			JFileChooser exploreur = new JFileChooser();
			int resultat = exploreur.showOpenDialog(this);
			if (resultat == JFileChooser.APPROVE_OPTION) {
				File file = exploreur.getSelectedFile();
				String path = file.getAbsolutePath();
				try {
					EngineVideo.executer(path, this.modelChoice);
				}
				catch (Exception exc) {
					JOptionPane.showMessageDialog(this,"Error Video Launch" + exc.getMessage());
				}
			}
		});
		//
		btnWebcam.addActionListener(e -> {
			//redirect In RealTime 
			System.out.println("Source : Webcam en cours avec le CNN " + this.modelChoice);
			try {
				EngineWebcam.executer(this.modelChoice);
			}
			catch (Exception exc){
				JOptionPane.showMessageDialog(this, "Error Webcam Launch" + exc.getMessage());
			}
		});
		//
		btnRetour.addActionListener(e -> {
			// back
			f.displayCNNScreen();
			});
	}
}
