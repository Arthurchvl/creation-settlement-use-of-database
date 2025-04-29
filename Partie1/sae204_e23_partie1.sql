CREATE SCHEMA "partie1";

SET SCHEMA 'partie1';

CREATE TABLE _individu (
	id_individu INT PRIMARY KEY,
	nom VARCHAR,
	prenom VARCHAR,
	date_naissance DATE,
	code_postal VARCHAR,
	ville VARCHAR,
	sexe CHAR,
	nationalite VARCHAR,
	INE VARCHAR
);

CREATE TABLE _etudiant (
	code_nip VARCHAR PRIMARY KEY,
	cat_socio_etu VARCHAR,
	cat_socio_parent VARCHAR,
	bourse_superieur BOOLEAN,
	mention_bac VARCHAR,
	serie_bac VARCHAR,
	dominante_bac VARCHAR,
	specialite_bac VARCHAR,
	mois_annee_obtention_bac VARCHAR(7),
	id_individu INT,
	FOREIGN KEY (id_individu) REFERENCES individu(id_individu) 
);

CREATE TABLE _candidat(
	no_candidat INT PRIMARY KEY,
	classement VARCHAR,
	boursier_lycee VARCHAR,
	profil_candidat VARCHAR,
	etablissement VARCHAR,
	dept_etablissement VARCHAR,
	ville_etablissement VARCHAR,
	niveau_etude VARCHAR,
	type_formation VARCHAR,
	serie_prec VARCHAR,
	dominante_prec VARCHAR,
	specialite_prec VARCHAR,
	lv1 VARCHAR,
	lv2 VARCHAR
);

CREATE TABLE _semestre (
	id_semestre INT PRIMARY KEY,
	num_semestre VARCHAR(5),
	annee_univ VARCHAR(9)
);

CREATE TABLE _module (
	id_module VARCHAR(5) PRIMARY KEY,
	libelle_module VARCHAR,
	ue VARCHAR(2)
);